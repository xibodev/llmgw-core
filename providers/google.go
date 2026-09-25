package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// Google's generative models are reachable through two deployments that share
// a request grammar but almost nothing else:
//
//	ai_studio  generativelanguage.googleapis.com/v1beta/models/{model}
//	           billed against Gemini API prepay credits
//	vertex_ai  {location-}aiplatform.googleapis.com/v1/projects/{project}/
//	           locations/{location}/publishers/google/models/{model}
//	           billed to the Cloud project's billing account
//
// Their catalogs differ too, and a model's locations differ per model, so a
// product configures one instance per deployment rather than sharing one.
const (
	googleAIStudioDefaultBase = "https://generativelanguage.googleapis.com/v1beta"
	vertexDefaultHost         = "aiplatform.googleapis.com"
	vertexDefaultAPI          = "v1"
	// Vertex AI's multi-region "global" endpoint carries the widest model
	// selection; regional endpoints are opt-in per model.
	vertexDefaultLocation = "global"
	googleDefaultTimeout  = 120 * time.Second
)

// GoogleDeployment selects the Google API a Google provider calls.
type GoogleDeployment string

const (
	// GoogleAIStudio is the Gemini API of Google AI Studio.
	GoogleAIStudio GoogleDeployment = "ai_studio"
	// GoogleVertexAI is Vertex AI in a Google Cloud project.
	GoogleVertexAI GoogleDeployment = "vertex_ai"
)

// GoogleConfig configures Google. Deployment is required; every other field
// is optional, and Project, Location and RequestType apply to Vertex AI only.
type GoogleConfig struct {
	Deployment GoogleDeployment
	// BaseURL is the API root. Empty uses Google's: AI Studio's v1beta root,
	// or the v1 root of Location's Vertex AI endpoint. A Vertex AI BaseURL is
	// an inference root that ends in /v1, as the default does; model
	// discovery replaces that version with v1beta1.
	BaseURL string
	// Project is the Vertex AI project calls are billed to. Empty takes the
	// credential's, and a credential that names another project is refused.
	Project string
	// Location is the Vertex AI location. Empty is "global".
	Location string
	// RequestType is how Vertex AI accounts an invocation: empty or
	// "default" leaves it to Vertex AI, "paygo" asks for shared pay-as-you-go
	// capacity, and "dedicated" for Provisioned Throughput.
	RequestType string
	// Client performs every request. Nil uses a client that times out after
	// 120 seconds.
	Client *http.Client
	// Now reads Retry-After dates and stamps catalog discovery. Nil uses
	// time.Now.
	Now func() time.Time
}

// Google implements core.Provider for Google's Gemini models on AI Studio or
// Vertex AI, speaking their native generateContent grammar as the gateway
// does. It is the whole vertical: the Chat transport, the catalog and what
// its rows mean, and the credential each deployment takes.
//
// Chat Completions is its surface. Google converts a Chat request to Gemini
// contents itself, with the gateway's mapping: the text of the messages,
// max_tokens and temperature. Every other field is dropped and reported in
// Response.Losses, a material loss when it changes the answer, as tools do.
// Google does not stream: Stream refuses and permits failover. A
// translation.Adapter in front serves Messages and Responses over Chat.
//
// Each operation authenticates with the credential it is given: an API key
// travels in x-goog-api-key and never in a URL, and a token is the bearer.
// Vertex AI refuses to run without a credential and nothing is sent; AI
// Studio sends what it has, as the gateway does, for a proxy that holds the
// key.
type Google struct {
	deployment  GoogleDeployment
	baseURL     string
	project     string
	location    string
	requestType string
	client      *http.Client
	now         func() time.Time
}

var _ core.Provider = (*Google)(nil)

// NewGoogle returns a Google provider for one deployment.
func NewGoogle(config GoogleConfig) (*Google, error) {
	base := strings.TrimRight(strings.TrimSpace(config.BaseURL), "/")
	if base != "" {
		if parsed, err := url.Parse(base); err != nil || (parsed.Scheme != "https" && parsed.Scheme != "http") || parsed.Host == "" {
			return nil, errors.New("Google base URL must be an absolute HTTP URL")
		}
	}
	p := &Google{deployment: config.Deployment, baseURL: base, client: config.Client, now: config.Now}
	switch config.Deployment {
	case GoogleAIStudio:
		if p.baseURL == "" {
			p.baseURL = googleAIStudioDefaultBase
		}
	case GoogleVertexAI:
		p.project = strings.TrimSpace(config.Project)
		p.location = strings.TrimSpace(config.Location)
		if p.location == "" {
			p.location = vertexDefaultLocation
		}
		if !googlePathSegment(p.location) || (p.project != "" && !googlePathSegment(p.project)) {
			return nil, errors.New("Vertex AI project and location must be single URL path segments")
		}
		switch strings.ToLower(strings.TrimSpace(config.RequestType)) {
		case "", "default":
		case "paygo":
			p.requestType = "shared"
		case "dedicated":
			p.requestType = "dedicated"
		default:
			return nil, errors.New("Vertex AI request type must be default, paygo or dedicated")
		}
	default:
		return nil, fmt.Errorf("Google deployment must be %q or %q", GoogleAIStudio, GoogleVertexAI)
	}
	if p.client == nil {
		p.client = &http.Client{Timeout: googleDefaultTimeout}
	}
	if p.now == nil {
		p.now = time.Now
	}
	return p, nil
}

// NativeSurfaces reports Chat Completions for every model.
func (p *Google) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceChatCompletions}
}

// Invoke performs one Chat Completions request.
func (p *Google) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	if request.Surface != core.ModelSurfaceChatCompletions {
		return core.Response{}, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	return p.chat(ctx, request)
}

// Stream refuses every request, as the gateway does: Google's transport
// has no streaming yet. Nothing is sent, and the refusal permits failover
// to a target that streams.
func (p *Google) Stream(_ context.Context, request core.Request) (core.StreamIter, error) {
	if request.Surface != core.ModelSurfaceChatCompletions {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	return nil, &core.ProviderError{
		Message: p.label() + ": streaming is not implemented for this provider yet; use a non-streaming request",
		Class:   core.ProviderErrorUnsupported, Classification: core.ProviderErrorClassification{FailoverEligible: true},
	}
}

// label names the deployment in messages, as the gateway names it.
func (p *Google) label() string { return string(p.deployment) }

// vertexHost is the one region rule every Vertex AI endpoint shares: the
// global endpoint has no location prefix, regional ones do. Inference and
// discovery disagree on API version and path, so only the host is shared.
func vertexHost(location string) string {
	if location != vertexDefaultLocation {
		return location + "-" + vertexDefaultHost
	}
	return vertexDefaultHost
}

// modelURL builds the endpoint of one model action. Vertex AI needs a
// project and carries the location twice: in a regional host and in the
// resource path.
func (p *Google) modelURL(model, project, action string) (string, error) {
	model = strings.TrimPrefix(strings.TrimSpace(model), "models/")
	if model == "" {
		return "", &core.ProviderError{Message: "a model name is required", Class: core.ProviderErrorInvalidRequest}
	}
	if !googlePathSegment(model) {
		return "", &core.ProviderError{Message: p.label() + ": the model name is not a Google model ID", Class: core.ProviderErrorInvalidRequest}
	}
	if p.deployment == GoogleAIStudio {
		return fmt.Sprintf("%s/models/%s:%s", p.baseURL, model, action), nil
	}
	if project == "" {
		return "", core.NewConfigurationError("vertex_ai: project is required (set 'project' on the provider)", nil)
	}
	base := p.baseURL
	if base == "" {
		base = "https://" + vertexHost(p.location) + "/" + vertexDefaultAPI
	}
	return fmt.Sprintf("%s/projects/%s/locations/%s/publishers/google/models/%s:%s",
		base, project, p.location, model, action), nil
}

// googlePathSegment reports a value that stays one segment of a URL path.
// The gateway interpolates models, projects and locations unescaped, so a
// value that could reshape the URL is refused instead of sent.
func googlePathSegment(value string) bool {
	return value != "" && !strings.ContainsFunc(value, func(r rune) bool {
		return r <= ' ' || r == 0x7f || strings.ContainsRune(`/\?#%`, r)
	})
}

// googleJSONBody decodes a JSON object body. Numbers decode as float64, as
// the gateway decodes a request, so a value passed through reaches Google as
// the gateway sends it.
func googleJSONBody(request core.Request) (map[string]any, error) {
	mediaType, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || mediaType != core.ContentTypeJSON {
		return nil, &core.ProviderError{Message: "a Google request body must be JSON", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	decoder := json.NewDecoder(bytes.NewReader(request.Body))
	var body map[string]any
	err = decoder.Decode(&body)
	if err == nil {
		if _, trailing := decoder.Token(); trailing != io.EOF {
			err = errors.New("the body continues after its JSON object")
		}
	}
	if err != nil || body == nil {
		return nil, &core.ProviderError{Message: "the Google request body is not a JSON object", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	return body, nil
}

// post sends one model action and returns Google's decoded answer, as the
// gateway's doContext does: encoding/json encodes the body with its
// defaults, a refused answer is reported from Google's error envelope, and
// an answer that is not a nonempty JSON object is unusable.
func (p *Google) post(ctx context.Context, authorization googleAuthorization, endpoint string, body any) (map[string]any, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, &core.ProviderError{Message: "the " + p.label() + " request could not be encoded", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, core.NewConfigurationError("the "+p.label()+" request could not be created", err)
	}
	request.Header.Set("Content-Type", core.ContentTypeJSON)
	// Only a Vertex AI instance has a request type, and only an invocation
	// carries it.
	if p.requestType != "" {
		request.Header.Set("X-Vertex-AI-LLM-Request-Type", p.requestType)
	}
	authorization.apply(request.Header)
	response, err := p.client.Do(request)
	if err != nil {
		return nil, transportFailure(ctx, p.label()+" could not be reached", err)
	}
	defer response.Body.Close()
	raw, err := readInvocationResponseBody(ctx, response, p.label())
	if err != nil {
		return nil, err
	}
	var decoded map[string]any
	decodeErr := json.Unmarshal(raw, &decoded)
	if response.StatusCode >= http.StatusBadRequest {
		return nil, p.statusFailure(response, raw, decoded)
	}
	if decodeErr != nil || len(decoded) == 0 {
		return nil, unusableResponse(p.label()+": invalid JSON in upstream response", decodeErr)
	}
	return decoded, nil
}

// statusFailure reports an answer Google refused, naming the cause its error
// envelope gives, as the gateway names it: exhausted billing, a model the
// project or location cannot use and a rejected credential are different
// operator problems. The message names the status and Google's status code,
// never the upstream's text. The cause's diagnostic is the gateway's own
// message, redacted, for a product to log.
func (p *Google) statusFailure(response *http.Response, raw []byte, decoded map[string]any) *core.ProviderError {
	status := response.StatusCode
	message := strings.TrimSpace(string(raw))
	googleStatus := ""
	if envelope, ok := decoded["error"].(map[string]any); ok {
		if text, ok := envelope["message"].(string); ok && text != "" {
			message = text
		}
		if text, ok := envelope["status"].(string); ok {
			googleStatus = text
		}
	}
	cause := ""
	switch {
	case googleStatus == "RESOURCE_EXHAUSTED" && strings.Contains(strings.ToLower(message), "credit"):
		cause = "provider billing exhausted"
	case status == http.StatusNotFound:
		cause = "model not available to this project or location"
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		cause = "credential rejected"
	}
	summary := fmt.Sprintf("%s returned HTTP %d", p.label(), status)
	if codexErrorIdentifier.MatchString(googleStatus) {
		summary += " (" + googleStatus + ")"
	}
	detail := p.label() + ": " + message
	if cause != "" {
		summary += ": " + cause
		detail = p.label() + ": " + cause + " — " + message
	}
	classification := statusClassification(status, retryAfterDelay(response.Header.Get("Retry-After"), p.now()))
	return &core.ProviderError{
		Message:        summary,
		Class:          core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: status}).ErrorClass,
		Classification: classification,
		Cause: &InvocationError{
			Msg: sanitizeDiagnosticTextLimit(detail, diagnosticErrorLimit), Status: status,
			Retryable: classification.Retryable, FailoverEligible: classification.FailoverEligible,
			CircuitFailure: classification.CircuitFailure, RetryAfter: classification.RetryAfter,
		},
	}
}
