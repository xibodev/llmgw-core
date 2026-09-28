package providers

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"strings"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
)

// openAIAudioAccess is how a fixture request authenticates.
type openAIAudioAccess struct {
	registryID    string
	credential    *core.Credential
	authorization string
}

// openAIAudioAccesses are a key with a credential header, and keyless
// access to an anonymous registry entry.
func openAIAudioAccesses() map[string]openAIAudioAccess {
	return map[string]openAIAudioAccess{
		"keyed":   {"", &core.Credential{APIKey: "fixture-key", Headers: map[string]string{"OpenAI-Organization": "org-fixture"}}, "Bearer fixture-key"},
		"keyless": {"ovh_ai_endpoints", nil, ""},
	}
}

func newOpenAIAudioProvider(t *testing.T, fixture audioWireFixture, registryID string) (*openAIBackend, *OpenAICompatible) {
	t.Helper()
	backend, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) { writeAudioFixture(t, w, fixture) })
	return backend, newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) {
		config.RegistryID, config.Headers = registryID, map[string]string{"X-Fixture": "static"}
	})
}

// fixtureAnswer is the body a fixture's upstream answers with.
func fixtureAnswer(t *testing.T, fixture audioWireFixture) []byte {
	t.Helper()
	switch {
	case fixture.Response.BodyBase64 != "":
		body, err := base64.StdEncoding.DecodeString(fixture.Response.BodyBase64)
		if err != nil {
			t.Fatal(err)
		}
		return body
	case fixture.Response.BodyText != "":
		return []byte(fixture.Response.BodyText)
	}
	return fixture.Response.Body
}

// checkAudioCall checks what the upstream received beside the body: the
// endpoint, the transport's headers and the credential's.
func checkAudioCall(t *testing.T, calls []openAICall, fixture audioWireFixture, authorization string) openAICall {
	t.Helper()
	if len(calls) != 1 {
		t.Fatalf("upstream = %+v", calls)
	}
	call := calls[0]
	if call.method != fixture.Request.Method || call.path != fixture.Request.Path || call.authorization != authorization ||
		call.header.Get("Content-Type") != fixture.Request.ContentType || call.accept != "" || call.vision != "" ||
		call.header.Get("X-Fixture") != "static" {
		t.Fatalf("upstream = %s %s authorization=%q headers=%v", call.method, call.path, call.authorization, call.header)
	}
	if organization := call.header.Get("OpenAI-Organization"); (authorization != "") != (organization == "org-fixture") {
		t.Fatalf("credential headers = %v", call.header)
	}
	return call
}

// A transcription's multipart upload reaches the upstream byte for byte,
// with its own Content-Type, boundary and quoting included, with a key or
// without one, and the answer comes back as sent, in whatever format the
// upload's response_format asked for.
func TestOpenAICompatibleForwardsTheTranscriptionUploadByteForByte(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"transcription-json", "transcription-srt"} {
		fixture := loadAudioFixture(t, "openai_compatible", name)
		upload, err := base64.StdEncoding.DecodeString(fixture.Request.BodyBase64)
		if err != nil {
			t.Fatal(err)
		}
		for access, test := range openAIAudioAccesses() {
			t.Run(name+" "+access, func(t *testing.T) {
				t.Parallel()
				backend, provider := newOpenAIAudioProvider(t, fixture, test.registryID)
				response, err := provider.Invoke(context.Background(), core.Request{
					Surface: core.ModelSurfaceAudioTranscriptions, Model: fixture.Request.Model,
					Body: bytes.Clone(upload), ContentType: fixture.Request.ContentType, Credential: test.credential,
				})
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(response.Body, fixtureAnswer(t, fixture)) || response.ContentType != fixture.Response.ContentType || len(response.Losses) != 0 {
					t.Fatalf("response = %q %q, losses = %v", response.ContentType, response.Body, response.Losses)
				}
				call := checkAudioCall(t, backend.take(), fixture, test.authorization)
				if call.body != string(upload) {
					t.Fatalf("the upload changed on its way:\n%q\nwant\n%q", call.body, upload)
				}
				fields, fileName, file := openAIUploadParts(t, call.header.Get("Content-Type"), []byte(call.body))
				want, _ := base64.StdEncoding.DecodeString(fixture.Request.FileBase64)
				if fileName != fixture.Request.FileName || !bytes.Equal(file, want) || len(fields) != len(fixture.Request.Fields) {
					t.Fatalf("upload = %v %q %q", fields, fileName, file)
				}
				for field, value := range fixture.Request.Fields {
					if fields[field] != value {
						t.Fatalf("field %s = %q, want %q", field, fields[field], value)
					}
				}
			})
		}
	}
}

// openAIUploadParts reads the fields and the file of a multipart upload.
func openAIUploadParts(t *testing.T, contentType string, body []byte) (map[string]string, string, []byte) {
	t.Helper()
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		t.Fatal(err)
	}
	reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	fields, fileName, file := map[string]string{}, "", []byte(nil)
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			return fields, fileName, file
		}
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(part)
		if err != nil {
			t.Fatal(err)
		}
		if part.FormName() == "file" {
			fileName, file = part.FileName(), data
		} else {
			fields[part.FormName()] = string(data)
		}
	}
}

// Speech is the caller's JSON object with the request's model, as
// Responses is sent, and the audio comes back as sent with the upstream's
// content type, with a key or without one.
func TestOpenAICompatibleSendsSpeechWithTheRequestsModel(t *testing.T) {
	t.Parallel()
	fixture := loadAudioFixture(t, "openai_compatible", "speech-mp3")
	var want bytes.Buffer
	if err := json.Compact(&want, fixture.Request.Body); err != nil {
		t.Fatal(err)
	}
	body := `{"voice":"alloy","input":"Hello from the fixture.","model":"a-routed-alias","response_format":"mp3","speed":1.25}`
	for access, test := range openAIAudioAccesses() {
		t.Run(access, func(t *testing.T) {
			t.Parallel()
			backend, provider := newOpenAIAudioProvider(t, fixture, test.registryID)
			response, err := provider.Invoke(context.Background(), core.Request{
				Surface: core.ModelSurfaceAudioSpeech, Model: fixture.Request.Model,
				Body: []byte(body), ContentType: "application/json; charset=utf-8", Credential: test.credential,
			})
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(response.Body, fixtureAnswer(t, fixture)) || response.ContentType != fixture.Response.ContentType || len(response.Losses) != 0 {
				t.Fatalf("response = %q %q, losses = %v", response.ContentType, response.Body, response.Losses)
			}
			if call := checkAudioCall(t, backend.take(), fixture, test.authorization); call.body != want.String() {
				t.Fatalf("upstream body = %s, want %s", call.body, want.String())
			}
		})
	}
}

// A refusal on either audio surface is classified as a Chat refusal is: a
// rejected key ends the request, and a 429 is rate limited, retryable and
// keeps its Retry-After. The message names the status and the upstream's
// identifiers, never its text.
func TestOpenAICompatibleClassifiesAudioRefusalsAsChatRefusals(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		fixture, message string
		class            core.ProviderErrorClass
		classification   core.ProviderErrorClassification
	}{
		{
			"error-401", "OpenAI-compatible returned HTTP 401 (code=invalid_api_key, type=invalid_request_error)",
			core.ProviderErrorAuth, core.ProviderErrorClassification{StatusCode: http.StatusUnauthorized},
		},
		{
			"error-429", "OpenAI-compatible returned HTTP 429 (code=rate_limit_exceeded, type=requests)",
			core.ProviderErrorRateLimited, core.ProviderErrorClassification{
				StatusCode: http.StatusTooManyRequests, Retryable: true, FailoverEligible: true, CircuitFailure: true, RetryAfter: 20 * time.Second,
			},
		},
	} {
		fixture := loadAudioFixture(t, "openai_compatible", test.fixture)
		for _, request := range openAIAudioRequests() {
			t.Run(test.fixture+" "+string(request.Surface), func(t *testing.T) {
				t.Parallel()
				backend, provider := newOpenAIAudioProvider(t, fixture, "")
				_, err := provider.Invoke(context.Background(), request)
				var failure *core.ProviderError
				if !errors.As(err, &failure) || failure.Class != test.class || failure.Classification != test.classification ||
					failure.Message != test.message || core.ClassifyError(err) != test.classification {
					t.Fatalf("err = %#v", err)
				}
				var cause *InvocationError
				if !errors.As(err, &cause) || cause.Status != fixture.Response.Status || strings.Contains(cause.Msg, "sk-fixture-rejected") {
					t.Fatalf("cause = %#v", cause)
				}
				if calls := backend.take(); len(calls) != 1 {
					t.Fatalf("upstream = %+v", calls)
				}
			})
		}
	}
	// The rest of the gateway's status set, on both surfaces.
	for _, test := range []struct {
		status    int
		class     core.ProviderErrorClass
		transient bool
	}{
		{http.StatusForbidden, core.ProviderErrorForbidden, false},
		{http.StatusBadRequest, core.ProviderErrorUpstream, false},
		{http.StatusServiceUnavailable, core.ProviderErrorUpstream, true},
	} {
		_, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
			w.Header().Set("Retry-After", "7")
			w.WriteHeader(test.status)
			_, _ = io.WriteString(w, `{"error":{"code":"fixture_code","message":"private upstream text"}}`)
		})
		provider := newTestOpenAICompatible(t, server, nil)
		for _, request := range openAIAudioRequests() {
			_, err := provider.Invoke(context.Background(), request)
			want := core.ProviderErrorClassification{
				StatusCode: test.status, Retryable: test.transient, FailoverEligible: test.transient, CircuitFailure: test.transient, RetryAfter: 7 * time.Second,
			}
			var failure *core.ProviderError
			if !errors.As(err, &failure) || failure.Class != test.class || failure.Classification != want || strings.Contains(err.Error(), "private upstream text") {
				t.Fatalf("%d %s: err = %#v", test.status, request.Surface, err)
			}
		}
	}
}

// openAIAudioRequests are one valid request of each audio surface.
func openAIAudioRequests() []core.Request {
	var upload bytes.Buffer
	writer := multipart.NewWriter(&upload)
	part, _ := writer.CreateFormFile("file", "sample.wav")
	_, _ = part.Write([]byte("RIFFfixture"))
	_ = writer.WriteField("model", "whisper-1")
	_ = writer.Close()
	return []core.Request{
		{
			Surface: core.ModelSurfaceAudioTranscriptions, Model: "whisper-1", Body: upload.Bytes(),
			ContentType: writer.FormDataContentType(), Credential: &core.Credential{APIKey: "fixture-key"},
		},
		openAIRequest(core.ModelSurfaceAudioSpeech, "tts-1", `{"input":"hello","voice":"alloy"}`, &core.Credential{APIKey: "fixture-key"}),
	}
}

// A request no audio endpoint could take is refused before anything is
// sent, and so is a stream: the audio surfaces are native, so the refusal
// is no *core.SurfaceError, but it permits failover as Google's does.
func TestOpenAICompatibleRefusesAudioBeforeSending(t *testing.T) {
	t.Parallel()
	_, server := newOpenAIBackend(t, func(http.ResponseWriter, *http.Request, int) { t.Error("a refused request reached the upstream") })
	provider := newTestOpenAICompatible(t, server, nil)
	transcription, speech := openAIAudioRequests()[0], openAIAudioRequests()[1]
	with := func(request core.Request, change func(*core.Request)) core.Request {
		change(&request)
		return request
	}
	var failure *core.ProviderError
	for name, request := range map[string]core.Request{
		"JSON upload":        with(transcription, func(r *core.Request) { r.ContentType = core.ContentTypeJSON }),
		"no boundary":        with(transcription, func(r *core.Request) { r.ContentType = "multipart/form-data" }),
		"mixed upload":       with(transcription, func(r *core.Request) { r.ContentType = "multipart/mixed; boundary=b" }),
		"empty upload":       with(transcription, func(r *core.Request) { r.Body = nil }),
		"blank model":        with(transcription, func(r *core.Request) { r.Model = " " }),
		"text speech":        with(speech, func(r *core.Request) { r.ContentType = "text/plain" }),
		"array speech":       with(speech, func(r *core.Request) { r.Body = []byte(`[]`) }),
		"trailing speech":    with(speech, func(r *core.Request) { r.Body = []byte(`{} {}`) }),
		"blank speech model": with(speech, func(r *core.Request) { r.Model = "" }),
	} {
		if _, err := provider.Invoke(context.Background(), request); !errors.As(err, &failure) || failure.Class != core.ProviderErrorInvalidRequest {
			t.Fatalf("%s: err = %v", name, err)
		}
	}
	for _, request := range []core.Request{transcription, speech} {
		request.Credential = &core.Credential{Token: `{"private_key":"fixture"}`, TokenType: core.TokenTypeGCPServiceAccount}
		if _, err := provider.Invoke(context.Background(), request); !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration ||
			strings.Contains(err.Error(), "fixture") {
			t.Fatalf("%s credential: err = %v", request.Surface, err)
		}
		request.Credential = nil
		stream, err := provider.Stream(context.Background(), request)
		var surface *core.SurfaceError
		if stream != nil || !errors.As(err, &failure) || errors.As(err, &surface) || failure.Class != core.ProviderErrorUnsupported ||
			core.ClassifyError(err) != (core.ProviderErrorClassification{FailoverEligible: true}) ||
			err.Error() != "OpenAI-compatible: streaming is not implemented for "+string(request.Surface)+"; use a non-streaming request" {
			t.Fatalf("%s stream: %v, err = %v", request.Surface, stream, err)
		}
	}
}

// Bedrock's OpenAI-compatible endpoint has no audio API, so NewBedrock
// serves neither audio surface, and refuses both as surfaces it lacks.
func TestBedrockServesNoAudioSurface(t *testing.T) {
	t.Parallel()
	_, server := newOpenAIBackend(t, func(http.ResponseWriter, *http.Request, int) { t.Error("an audio request reached Bedrock") })
	provider, err := NewBedrock("", server.URL+"/openai/v1", OpenAICompatibleConfig{Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	if surfaces := provider.NativeSurfaces("fixture.model-v1"); len(surfaces) != 1 || surfaces[0] != core.ModelSurfaceChatCompletions {
		t.Fatalf("surfaces = %v", surfaces)
	}
	var surface *core.SurfaceError
	for _, request := range openAIAudioRequests() {
		_, invokeErr := provider.Invoke(context.Background(), request)
		_, streamErr := provider.Stream(context.Background(), request)
		if !errors.As(invokeErr, &surface) || !errors.As(streamErr, &surface) || core.PreservesWire(provider, request.Model, request.Surface) {
			t.Fatalf("%s: invoke err = %v, stream err = %v", request.Surface, invokeErr, streamErr)
		}
	}
}

// An answer is labelled with the upstream's content type, or as unknown
// binary when it states none, and speech without audio is unusable.
func TestOpenAICompatibleLabelsTheAudioAnswer(t *testing.T) {
	t.Parallel()
	for name, test := range map[string]struct {
		contentType, body, want string
		request                 core.Request
	}{
		"transcription text": {"text/plain; charset=utf-8", "Hello\n", "text/plain; charset=utf-8", openAIAudioRequests()[0]},
		"transcription vtt":  {"text/vtt", "WEBVTT\n\n00:00.000 --> 00:01.000\nHello\n", "text/vtt", openAIAudioRequests()[0]},
		"unlabelled speech":  {"", "OggS", openAIUnknownContentType, openAIAudioRequests()[1]},
	} {
		_, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
			// A nil Content-Type keeps the server from sniffing one.
			w.Header()["Content-Type"] = nil
			if test.contentType != "" {
				w.Header().Set("Content-Type", test.contentType)
			}
			_, _ = io.WriteString(w, test.body)
		})
		response, err := newTestOpenAICompatible(t, server, nil).Invoke(context.Background(), test.request)
		if err != nil || response.ContentType != test.want || string(response.Body) != test.body {
			t.Fatalf("%s: response = %q %q, err = %v", name, response.ContentType, response.Body, err)
		}
	}
	_, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) { w.Header().Set("Content-Type", "audio/mpeg") })
	_, err := newTestOpenAICompatible(t, server, nil).Invoke(context.Background(), openAIAudioRequests()[1])
	var failure *core.ProviderError
	if !errors.As(err, &failure) || failure.Class != core.ProviderErrorUpstream ||
		failure.Classification != (core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}) {
		t.Fatalf("empty speech: err = %#v", err)
	}
}

// oversizedAudio answers every request with more than the gateway reads of
// an inference answer, without a network.
type oversizedAudio struct{}

func (oversizedAudio) RoundTrip(*http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"audio/mpeg"}},
		Body: io.NopCloser(io.LimitReader(zeroReader{}, inferenceMaxResponseBytes+1)),
	}, nil
}

// An audio answer is bounded as every inference answer is. Not parallel:
// it reads the whole bound.
func TestOpenAICompatibleBoundsTheAudioAnswer(t *testing.T) {
	provider, err := NewOpenAICompatible(OpenAICompatibleConfig{BaseURL: "https://upstream.example.test/v1", Client: &http.Client{Transport: oversizedAudio{}}})
	if err != nil {
		t.Fatal(err)
	}
	for _, request := range openAIAudioRequests() {
		response, err := provider.Invoke(context.Background(), request)
		var failure *core.ProviderError
		if response.Body != nil || !errors.As(err, &failure) || failure.Class != core.ProviderErrorUpstream ||
			err.Error() != "the OpenAI-compatible response exceeds the size limit" {
			t.Fatalf("%s: err = %v", request.Surface, err)
		}
	}
}
