package providers

import (
	"errors"
	"fmt"
	"net/http"
	"strings"

	gcp "github.com/xibodev/llm-provider-auth/gcp"
	core "github.com/xibodev/llmgw-core"
)

// googleAccess is the credential material of one operation and, on Vertex
// AI, the project the operation is billed to.
type googleAccess struct {
	apiKey string
	bearer string
	// account is a service-account key, exchanged for the bearer only once
	// the operation is ready to be sent.
	account *gcp.Credential
	project string
}

// googleAuthorization is the header one operation authenticates with. The
// zero value sends none.
type googleAuthorization struct{ name, value string }

func (a googleAuthorization) apply(header http.Header) {
	if a.name != "" {
		header.Set(a.name, a.value)
	}
}

// bearer reports an OAuth principal, which Vertex AI model discovery needs.
func (a googleAuthorization) bearer() bool { return a.name == "Authorization" }

// access reads a request's credential as the gateway reads a Google
// connection: a token is an OAuth bearer, and an API key travels in
// x-goog-api-key. A bearer wins when a credential holds both, because
// Vertex AI refuses a request that presents both. On Vertex AI a
// service-account key is parsed, and its project fills or checks the
// configured one. A credential of a kind Google does not serve is refused,
// so its material is never sent as a bearer.
//
// Vertex AI needs a credential: the gateway fails without one rather than
// send a request Vertex AI answers with a 401, which reads as a rejected
// key instead of an absent one.
func (p *Google) access(credential *core.Credential) (googleAccess, error) {
	var access googleAccess
	credentialProject, source := "", "credential"
	if credential != nil {
		credentialProject = strings.TrimSpace(credential.Metadata[core.CredentialMetadataProjectID])
		access.apiKey = strings.TrimSpace(credential.APIKey)
		switch kind := strings.TrimSpace(credential.TokenType); {
		case strings.EqualFold(kind, core.TokenTypeAPIKey):
			if access.apiKey == "" {
				access.apiKey = strings.TrimSpace(credential.Token)
			}
		case strings.EqualFold(kind, core.TokenTypeGCPServiceAccount) && p.deployment == GoogleVertexAI:
			account, err := gcp.Parse([]byte(credential.Token))
			if err != nil {
				// The parser describes the key's shape, never its contents.
				return googleAccess{}, core.NewConfigurationError("vertex_ai: "+err.Error(), err)
			}
			access.account = account
			// A project in the metadata is the one calls are billed to when
			// it is not the key's own.
			if credentialProject == "" {
				credentialProject, source = account.ProjectID(), "service account"
			}
		case strings.EqualFold(kind, core.TokenTypeGCPServiceAccount), strings.EqualFold(kind, core.TokenTypeAnthropicSetupToken):
			return googleAccess{}, core.NewConfigurationError(fmt.Sprintf("%s needs an API key or an access token, not a %s credential", p.label(), kind), nil)
		default:
			access.bearer = strings.TrimSpace(credential.Token)
		}
	}
	if p.deployment != GoogleVertexAI {
		return access, nil
	}
	if access.apiKey == "" && access.bearer == "" && access.account == nil {
		return googleAccess{}, core.NewConfigurationError("vertex_ai: no credential configured — vertex_ai needs an API key, an access token "+
			"or a Google service account key; add one before sending requests", nil)
	}
	project, err := p.vertexProject(credentialProject, source)
	if err != nil {
		return googleAccess{}, err
	}
	access.project = project
	return access, nil
}

// vertexProject is the project a Vertex AI operation is billed to, as the
// gateway resolves it: the configured project, filled in from the
// credential's when unset. When both are set and differ, the operation
// fails here rather than as an opaque 403 from Google.
func (p *Google) vertexProject(credentialProject, source string) (string, error) {
	project := p.project
	switch {
	case credentialProject == "":
	case project == "":
		project = credentialProject
	case !strings.EqualFold(project, credentialProject):
		return "", core.NewConfigurationError(fmt.Sprintf(
			"vertex_ai: configured project %q does not match the %s project %q", project, source, credentialProject,
		), nil)
	}
	if project != "" && !googlePathSegment(project) {
		return "", core.NewConfigurationError(fmt.Sprintf("vertex_ai: project %q is not a Google Cloud project ID", project), nil)
	}
	return project, nil
}

// authorization is the header an operation authenticates with. A
// service-account key is exchanged for a cloud-platform token through the
// instance's cache, which mints again shortly before a token expires.
func (p *Google) authorization(access googleAccess) (googleAuthorization, error) {
	bearer := access.bearer
	if access.account != nil {
		token, err := p.tokens.AccessToken(access.account, gcp.CloudPlatformScope)
		if err != nil {
			return googleAuthorization{}, googleTokenFailure(p.label(), err)
		}
		bearer = strings.TrimSpace(token)
	}
	switch {
	case bearer != "":
		return googleAuthorization{name: "Authorization", value: "Bearer " + bearer}, nil
	case access.apiKey != "":
		return googleAuthorization{name: "x-goog-api-key", value: access.apiKey}, nil
	}
	return googleAuthorization{}, nil
}

// googleTokenFailure reports a service-account token exchange that failed,
// classified as the gateway classifies it: a refusal from the token
// endpoint permits failover unless it is 401 or 403, and a transient one may
// repeat; an endpoint out of reach may repeat; any other failure permits
// failover. Only a refusal keeps its status, so an answer without a token
// never reads as a healthy upstream. The message never quotes the endpoint.
// Its *gcp.TokenError, which the auth library redacts, stays in the cause.
func googleTokenFailure(label string, err error) *core.ProviderError {
	message := label + ": service account token refresh failed"
	cause := &InvocationError{Msg: message, FailoverEligible: true, Cause: err}
	failure := &core.ProviderError{
		Message: message, Class: core.ProviderErrorAuth, Cause: cause,
		Classification: core.ProviderErrorClassification{FailoverEligible: true},
	}
	var exchange *gcp.TokenError
	switch {
	case errors.As(err, &exchange) && exchange.StatusCode >= http.StatusBadRequest:
		status := exchange.StatusCode
		transient := statusClassification(status, 0)
		failure.Message = fmt.Sprintf("%s (HTTP %d)", message, status)
		if transient.Retryable {
			failure.Class = core.ClassifyProviderFailure(core.ProviderFailure{StatusCode: status}).ErrorClass
		}
		failure.Classification = core.ProviderErrorClassification{
			StatusCode: status, Retryable: transient.Retryable, CircuitFailure: transient.CircuitFailure,
			FailoverEligible: status != http.StatusUnauthorized && status != http.StatusForbidden,
		}
		cause.Status, cause.Retryable, cause.CircuitFailure = status, transient.Retryable, transient.CircuitFailure
		cause.FailoverEligible = failure.Classification.FailoverEligible
	case errors.As(err, &exchange) && exchange.Code == "transport":
		failure.Class = core.ProviderErrorTransport
		failure.Classification = core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}
		cause.Retryable, cause.CircuitFailure = true, true
	}
	return failure
}
