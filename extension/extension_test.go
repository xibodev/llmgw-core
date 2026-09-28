package extension_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/extension"
	"github.com/xibodev/llmgw-core/oauthflow"
)

const secret = "s3cret"

// daemon is a fake extension daemon: every route answers through handle,
// after the shared secret is checked.
func daemon(t *testing.T, handle func(w http.ResponseWriter, r *http.Request)) (*extension.Client, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if got := r.Header.Get("Authorization"); got != "Bearer "+secret {
			http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
			return
		}
		handle(w, r)
	}))
	t.Cleanup(server.Close)
	client, err := extension.NewClient(extension.Config{BaseURL: server.URL + "/", Secret: secret})
	if err != nil {
		t.Fatal(err)
	}
	return client, &calls
}

func writeJSON(t *testing.T, w http.ResponseWriter, status int, value any) {
	t.Helper()
	w.Header().Set("Content-Type", core.ContentTypeJSON)
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		t.Error(err)
	}
}

func TestNewClientValidatesBaseURL(t *testing.T) {
	for _, base := range []string{
		"", "127.0.0.1:18888", "ftp://127.0.0.1", "http://", "http://user:pw@127.0.0.1",
		"http://127.0.0.1?x=1", "http://127.0.0.1#fragment",
	} {
		if _, err := extension.NewClient(extension.Config{BaseURL: base}); err == nil {
			t.Errorf("NewClient(%q) accepted an invalid base URL", base)
		}
	}
	for _, base := range []string{"http://127.0.0.1:18888", "https://daemon.internal/proxy/"} {
		if _, err := extension.NewClient(extension.Config{BaseURL: base}); err != nil {
			t.Errorf("NewClient(%q) = %v", base, err)
		}
	}
}

func TestInfoDecodesProvidersAndCredentialKinds(t *testing.T) {
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/extension/v1/info" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = io.WriteString(w, `{"version":"1.2.0","providers":[
			{"id":"oauth_chat","name":"Signed-in chat","surfaces":["chat_completions","responses"],"has_oauth":true,"has_refresh":true,"credential":"oauth","oauth_methods":["device","manual"]},
			{"id":"pasted","name":"Pasted token","surfaces":["messages"],"credential":"token"},
			{"id":"legacy_oauth","name":"Old daemon","surfaces":["chat_completions"],"has_oauth":true},
			{"id":"legacy_plain","name":"Old daemon","surfaces":["audio_speech"]}
		]}`)
	})
	info, err := client.Info(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if info.Version != "1.2.0" || len(info.Providers) != 4 {
		t.Fatalf("info = %+v", info)
	}
	signedIn := info.Providers[0]
	if !reflect.DeepEqual(signedIn.Surfaces, []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses}) ||
		!reflect.DeepEqual(signedIn.OAuthMethods, []oauthflow.Method{oauthflow.MethodDevice, oauthflow.MethodManual}) {
		t.Fatalf("signed-in provider = %+v", signedIn)
	}
	want := []extension.CredentialKind{extension.CredentialOAuth, extension.CredentialToken, extension.CredentialOAuth, ""}
	for index, provider := range info.Providers {
		if got := provider.CredentialKind(); got != want[index] {
			t.Errorf("%s CredentialKind() = %q, want %q", provider.ID, got, want[index])
		}
	}
}

func TestClientWithoutSecretSendsNoAuthorization(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := r.Header["Authorization"]; ok {
			t.Error("a client without a secret sent Authorization")
		}
		_, _ = io.WriteString(w, `{"version":"1","providers":[]}`)
	}))
	defer server.Close()
	client, err := extension.NewClient(extension.Config{BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Info(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestInvokeCarriesTheOperationAndItsCredential(t *testing.T) {
	losses := []core.Loss{{Path: "reasoning", Detail: "dropped"}}
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/extension/v1/oauth_chat/invoke" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		for header, want := range map[string]string{
			extension.HeaderSurface:           "chat_completions",
			extension.HeaderModel:             "gpt-test",
			"Content-Type":                    core.ContentTypeJSON,
			extension.HeaderCredentialToken:   "access-token",
			extension.HeaderCredentialAccount: "account-1",
			extension.HeaderCredentialType:    "Bearer",
		} {
			if got := r.Header.Get(header); got != want {
				t.Errorf("%s = %q, want %q", header, got, want)
			}
		}
		credential := extension.CredentialFromHeaders(r.Header)
		if credential == nil || credential.Metadata["project"] != "p-1" {
			t.Errorf("credential from headers = %v", credential)
		}
		body, _ := io.ReadAll(r.Body)
		extension.SetLossesHeader(w.Header(), losses)
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = w.Write(append([]byte(`{"echo":`), append(body, '}')...))
	})
	response, err := client.Invoke(context.Background(), "oauth_chat", core.Request{
		Surface:     core.ModelSurfaceChatCompletions,
		Model:       "gpt-test",
		Body:        []byte(`{"model":"gpt-test"}`),
		ContentType: core.ContentTypeJSON,
		Credential: &core.Credential{
			Token: "access-token", AccountID: "account-1", TokenType: "Bearer",
			Metadata: map[string]string{"project": "p-1"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(response.Body) != `{"echo":{"model":"gpt-test"}}` || response.ContentType != core.ContentTypeJSON {
		t.Fatalf("response = %s (%s)", response.Body, response.ContentType)
	}
	if !reflect.DeepEqual(response.Losses, losses) {
		t.Fatalf("losses = %+v, want %+v", response.Losses, losses)
	}
}

func TestCredentialHeadersRoundTrip(t *testing.T) {
	header := http.Header{}
	extension.SetCredentialHeaders(header, nil)
	if len(header) != 0 || extension.CredentialFromHeaders(header) != nil {
		t.Fatalf("a nil credential set headers %v", header)
	}

	extension.SetCredentialHeaders(header, &core.Credential{APIKey: "key-1"})
	if header.Get(extension.HeaderCredentialToken) != "key-1" || header.Get(extension.HeaderCredentialType) != core.TokenTypeAPIKey {
		t.Fatalf("API key headers = %v", header)
	}
	if credential := extension.CredentialFromHeaders(header); credential.APIKey != "key-1" || credential.Token != "" {
		t.Fatalf("API key credential = %v", credential)
	}

	header = http.Header{}
	extension.SetCredentialHeaders(header, &core.Credential{Token: "pasted", TokenType: "pasted_token"})
	if credential := extension.CredentialFromHeaders(header); credential.Token != "pasted" || credential.APIKey != "" {
		t.Fatalf("token credential = %v", credential)
	}

	header = http.Header{}
	header.Set(extension.HeaderCredentialMetadata, "not base64!")
	header.Set(extension.HeaderCredentialToken, "t")
	if credential := extension.CredentialFromHeaders(header); credential.Token != "t" || credential.Metadata != nil {
		t.Fatalf("undecodable metadata = %v", credential)
	}
}

func TestStreamRelaysRecordsThatCarryData(t *testing.T) {
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/extension/v1/stream_chat/stream" || r.Header.Get("Accept") != core.ContentTypeEventStream {
			t.Errorf("request = %s, Accept %q", r.URL.Path, r.Header.Get("Accept"))
		}
		extension.SetLossesHeader(w.Header(), []core.Loss{{Path: "logprobs"}})
		w.Header().Set("Content-Type", core.ContentTypeEventStream)
		_, _ = io.WriteString(w, ": keepalive\n\n"+
			"data: {\"a\":1}\n\n"+
			"event: ping\n\n"+
			"data:\n\n"+
			"event: delta\r\ndata: {\"b\":2}\r\n\r\n"+
			"data: [DONE]\n\n"+
			"data: tail")
	})
	stream, err := client.Stream(context.Background(), "stream_chat", core.Request{
		Surface: core.ModelSurfaceChatCompletions, Model: "gpt-test", Body: []byte(`{}`), ContentType: core.ContentTypeJSON,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var frames []string
	for {
		frame, err := stream.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		frames = append(frames, string(frame))
	}
	want := []string{"data: {\"a\":1}\n\n", "event: delta\r\ndata: {\"b\":2}\r\n\r\n", "data: [DONE]\n\n", "data: tail\n\n"}
	if !reflect.DeepEqual(frames, want) {
		t.Fatalf("frames = %q\nwant     %q", frames, want)
	}
	if losses := core.StreamLosses(stream); len(losses) != 1 || losses[0].Path != "logprobs" {
		t.Fatalf("stream losses = %+v", losses)
	}
}

func TestStreamRecordOverTheLimitEndsTheRequest(t *testing.T) {
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", core.ContentTypeEventStream)
		_, _ = io.WriteString(w, "data: "+strings.Repeat("x", 4<<20)+"\n\n")
	})
	stream, err := client.Stream(context.Background(), "p", core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	_, err = stream.Next()
	var providerError *core.ProviderError
	if !errors.As(err, &providerError) || providerError.Class != core.ProviderErrorUpstream || core.ClassifyError(err).Disposition() != core.DispositionTerminal {
		t.Fatalf("oversized record error = %#v", err)
	}
}

func TestFailedOperationsAreClassifiedLikeUpstreamFailures(t *testing.T) {
	for _, test := range []struct {
		name        string
		status      int
		retryAfter  string
		body        string
		class       core.ProviderErrorClass
		disposition core.Disposition
		message     string
		hidden      string
	}{
		{name: "rate limited", status: http.StatusTooManyRequests, retryAfter: "7",
			body: `{"error":{"message":"slow down","code":429}}`, class: core.ProviderErrorRateLimited,
			disposition: core.DispositionRetryable, message: "slow down"},
		{name: "string error", status: http.StatusBadRequest, body: `{"error":"model not found"}`,
			class: core.ProviderErrorUpstream, disposition: core.DispositionTerminal, message: "model not found"},
		{name: "unauthorized", status: http.StatusUnauthorized, body: `{"error":{"message":"token expired"}}`,
			class: core.ProviderErrorAuth, disposition: core.DispositionTerminal, message: "token expired"},
		{name: "raw body", status: http.StatusBadGateway, body: "upstream said something private",
			class: core.ProviderErrorUpstream, disposition: core.DispositionRetryable, message: "(HTTP 502)", hidden: "private"},
	} {
		t.Run(test.name, func(t *testing.T) {
			client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
				if test.retryAfter != "" {
					w.Header().Set("Retry-After", test.retryAfter)
				}
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, test.body)
			})
			_, err := client.Invoke(context.Background(), "p", core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "m"})
			var providerError *core.ProviderError
			if !errors.As(err, &providerError) {
				t.Fatalf("error = %#v", err)
			}
			classification := core.ClassifyError(err)
			if providerError.Class != test.class || classification.Disposition() != test.disposition || classification.StatusCode != test.status {
				t.Fatalf("class %q, classification %+v", providerError.Class, classification)
			}
			if !strings.Contains(err.Error(), test.message) || (test.hidden != "" && strings.Contains(err.Error(), test.hidden)) {
				t.Fatalf("message = %q", err.Error())
			}
			if test.retryAfter != "" && classification.RetryAfter != 7*time.Second {
				t.Fatalf("RetryAfter = %v", classification.RetryAfter)
			}
		})
	}
}

func TestUnansweredRequestsAreRetryableUnlessCanceled(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	base := server.URL
	server.Close()
	client, err := extension.NewClient(extension.Config{BaseURL: base, Secret: secret})
	if err != nil {
		t.Fatal(err)
	}
	request := core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "m"}
	_, err = client.Invoke(context.Background(), "p", request)
	var providerError *core.ProviderError
	if !errors.As(err, &providerError) || providerError.Class != core.ProviderErrorTransport || core.ClassifyError(err).Disposition() != core.DispositionRetryable {
		t.Fatalf("unanswered error = %#v", err)
	}
	if strings.Contains(err.Error(), base) {
		t.Fatalf("message names the daemon's address: %q", err.Error())
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = client.Invoke(ctx, "p", request)
	if core.ClassifyError(err).Disposition() != core.DispositionTerminal {
		t.Fatalf("canceled error classification = %+v", core.ClassifyError(err))
	}
}

func TestListModelsSendsTheCredential(t *testing.T) {
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/extension/v1/catalog_chat/models" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get(extension.HeaderCredentialToken) != "t" {
			t.Errorf("token = %q", r.Header.Get(extension.HeaderCredentialToken))
		}
		writeJSON(t, w, http.StatusOK, extension.ModelsResponse{Models: []core.ModelInfo{{ID: "model-test", Object: "model"}}})
	})
	models, err := client.ListModels(context.Background(), "catalog_chat", &core.Credential{Token: "t"})
	if err != nil || len(models) != 1 || models[0].ID != "model-test" {
		t.Fatalf("models = %+v, %v", models, err)
	}
}

func TestInvalidAnswersAreUpstreamFailures(t *testing.T) {
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "not json")
	})
	_, err := client.ListModels(context.Background(), "p", nil)
	var providerError *core.ProviderError
	if !errors.As(err, &providerError) || providerError.Class != core.ProviderErrorUpstream {
		t.Fatalf("invalid answer error = %#v", err)
	}
}

func TestRefresh(t *testing.T) {
	var answer func(w http.ResponseWriter)
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/extension/v1/oauth_chat/refresh" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		// The record keeps the field names both products already exchange.
		var raw map[string]map[string]any
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil || raw["record"]["RefreshToken"] != "refresh-1" {
			t.Errorf("refresh request = %v, %v", raw, err)
		}
		answer(w)
	})
	current := tokenstore.Record{AccessToken: "access-1", RefreshToken: "refresh-1", AccountID: "account-1"}

	answer = func(w http.ResponseWriter) {
		writeJSON(t, w, http.StatusOK, extension.RefreshResponse{Record: tokenstore.Record{AccessToken: "access-2", RefreshToken: "refresh-2", AccountID: "account-1"}})
	}
	refreshed, err := client.Refresh(context.Background(), "oauth_chat", current)
	if err != nil || refreshed.AccessToken != "access-2" || refreshed.RefreshToken != "refresh-2" {
		t.Fatalf("refreshed = %v, %v", refreshed, err)
	}

	answer = func(w http.ResponseWriter) {
		writeJSON(t, w, http.StatusBadRequest, extension.RefreshResponse{Terminal: true, Error: "invalid_grant"})
	}
	_, err = client.Refresh(context.Background(), "oauth_chat", current)
	var refreshError *extension.RefreshError
	if !errors.As(err, &refreshError) || !tokenstore.IsTerminal(err) || !strings.Contains(err.Error(), "invalid_grant") {
		t.Fatalf("terminal refresh error = %#v", err)
	}

	answer = func(w http.ResponseWriter) {
		http.Error(w, `{"error":"provider does not support refresh"}`, http.StatusBadRequest)
	}
	_, err = client.Refresh(context.Background(), "oauth_chat", current)
	if tokenstore.IsTerminal(err) || !strings.Contains(err.Error(), "does not support refresh") {
		t.Fatalf("refused refresh error = %v", err)
	}
}

func TestRefreshFuncServesTheCoordinator(t *testing.T) {
	var refreshes atomic.Int32
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		refreshes.Add(1)
		writeJSON(t, w, http.StatusOK, extension.RefreshResponse{Record: tokenstore.Record{
			AccessToken: "access-2", RefreshToken: "refresh-2", AccountID: "account-1", Expiry: time.Now().Add(time.Hour),
		}})
	})
	store := tokenstore.NewMemory()
	ctx := context.Background()
	if _, err := store.Save(ctx, "signed-in", tokenstore.Record{
		AccessToken: "access-1", RefreshToken: "refresh-1", AccountID: "account-1", Expiry: time.Now().Add(-time.Minute),
	}); err != nil {
		t.Fatal(err)
	}
	coordinator, err := tokenstore.NewCoordinator(store, client.RefreshFunc("oauth_chat"))
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		record, err := coordinator.Token(ctx, "signed-in")
		if err != nil || record.AccessToken != "access-2" {
			t.Fatalf("token = %v, %v", record, err)
		}
	}
	if refreshes.Load() != 1 {
		t.Fatalf("refreshes = %d, want 1", refreshes.Load())
	}
}

func TestOAuthDriver(t *testing.T) {
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		var raw map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			t.Errorf("decode %s: %v", r.URL.Path, err)
		}
		switch r.URL.Path {
		case "/extension/v1/oauth_chat/oauth/start":
			// The daemon reads snake_case start fields.
			var start map[string]any
			_ = json.Unmarshal(mustJSON(t, raw), &start)
			if start["method"] != "device" || start["redirect_uri"] != "https://product/callback" ||
				start["params"].(map[string]any)["client_id"] != "client-1" {
				t.Errorf("start request = %v", start)
			}
			writeJSON(t, w, http.StatusOK, extension.OAuthStartResponse{Authorization: oauthflow.Authorization{
				UserCode: "ABCD-1234", VerificationURI: "https://provider/device", Interval: 5 * time.Second,
				Secrets: oauthflow.Secrets{DeviceCode: "device-1"},
			}})
		case "/extension/v1/oauth_chat/oauth/poll":
			var poll extension.OAuthPollRequest
			_ = json.Unmarshal(mustJSON(t, raw), &poll)
			if poll.Flow.Secrets.DeviceCode != "device-1" {
				t.Errorf("poll flow secrets = %+v", poll.Flow.Secrets)
			}
			writeJSON(t, w, http.StatusOK, extension.OAuthPollResponse{Result: oauthflow.PollResult{
				Status: oauthflow.PollApproved, Record: tokenstore.Record{AccessToken: "access-1"},
			}})
		case "/extension/v1/oauth_chat/oauth/exchange":
			var exchange extension.OAuthExchangeRequest
			_ = json.Unmarshal(mustJSON(t, raw), &exchange)
			if exchange.Code != "code-1" || exchange.Flow.Secrets.Verifier != "verifier-1" {
				t.Errorf("exchange request = %+v", exchange)
			}
			writeJSON(t, w, http.StatusOK, extension.OAuthExchangeResponse{Record: tokenstore.Record{AccessToken: "access-2"}})
		case "/extension/v1/refuses/oauth/start":
			writeJSON(t, w, http.StatusBadRequest, extension.OAuthStartResponse{Error: "method manual is not supported"})
		case "/extension/v1/breaks/oauth/poll":
			writeJSON(t, w, http.StatusInternalServerError, extension.ErrorResponse{Error: extension.ErrorDetail{Message: "provider unavailable", Code: 500}})
		default:
			t.Errorf("unexpected route %s", r.URL.Path)
		}
	})
	ctx := context.Background()
	driver := client.OAuthDriver("oauth_chat")

	authorization, err := driver.Start(ctx, oauthflow.StartRequest{
		Method: oauthflow.MethodDevice, RedirectURI: "https://product/callback", Params: map[string]string{"client_id": "client-1"},
	})
	if err != nil || authorization.UserCode != "ABCD-1234" || authorization.Interval != 5*time.Second || authorization.Secrets.DeviceCode != "device-1" {
		t.Fatalf("authorization = %+v, %v", authorization, err)
	}
	result, err := driver.Poll(ctx, oauthflow.Flow{Secrets: authorization.Secrets})
	if err != nil || result.Status != oauthflow.PollApproved || result.Record.AccessToken != "access-1" {
		t.Fatalf("poll = %+v, %v", result, err)
	}
	record, err := driver.Exchange(ctx, oauthflow.Flow{Secrets: oauthflow.Secrets{Verifier: "verifier-1"}}, "code-1")
	if err != nil || record.AccessToken != "access-2" {
		t.Fatalf("exchange = %v, %v", record, err)
	}

	if _, err := client.OAuthDriver("refuses").Start(ctx, oauthflow.StartRequest{Method: oauthflow.MethodManual}); err == nil ||
		!strings.Contains(err.Error(), "method manual is not supported") {
		t.Fatalf("refused start error = %v", err)
	}
	if _, err := client.OAuthDriver("breaks").Poll(ctx, oauthflow.Flow{}); err == nil || !strings.Contains(err.Error(), "provider unavailable") {
		t.Fatalf("failed poll error = %v", err)
	}
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestProviderServesOnlyItsSurfaces(t *testing.T) {
	client, calls := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/extension/v1/speech/invoke" {
			t.Errorf("route = %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = io.WriteString(w, "mp3")
	})
	provider := extension.NewProvider(client, extension.ProviderInfo{ID: "speech", Surfaces: []core.ModelSurface{core.ModelSurfaceAudioSpeech}})
	surfaces := provider.NativeSurfaces("any")
	surfaces[0] = core.ModelSurfaceImages
	if !core.ServesNatively(provider, "voice", core.ModelSurfaceAudioSpeech) {
		t.Fatal("NativeSurfaces shares its slice with the caller")
	}

	_, err := provider.Invoke(context.Background(), core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "voice"})
	var surfaceError *core.SurfaceError
	if !errors.As(err, &surfaceError) || calls.Load() != 0 {
		t.Fatalf("unsupported surface error = %v after %d calls", err, calls.Load())
	}
	response, err := provider.Invoke(context.Background(), core.Request{Surface: core.ModelSurfaceAudioSpeech, Model: "voice", Body: []byte("hi"), ContentType: "text/plain"})
	if err != nil || string(response.Body) != "mp3" || response.ContentType != "audio/mpeg" {
		t.Fatalf("speech = %q (%s), %v", response.Body, response.ContentType, err)
	}
}

// catalogProvider is the provider "catalog" of a daemon whose models route
// answers with rows once it has checked the route and the credential.
func catalogProvider(t *testing.T, surfaces []core.ModelSurface, rows []core.ModelInfo) *extension.Provider {
	t.Helper()
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/extension/v1/catalog/models" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if token, kind := r.Header.Get(extension.HeaderCredentialToken), r.Header.Get(extension.HeaderCredentialType); token != "key-1" || kind != core.TokenTypeAPIKey {
			t.Errorf("credential = %q (%s)", token, kind)
		}
		writeJSON(t, w, http.StatusOK, extension.ModelsResponse{Models: rows})
	})
	return extension.NewProvider(client, extension.ProviderInfo{ID: "catalog", Surfaces: surfaces})
}

func listModels(t *testing.T, provider *extension.Provider) []core.ModelInfo {
	t.Helper()
	models, err := provider.ListModels(context.Background(), &core.Credential{APIKey: "key-1"})
	if err != nil {
		t.Fatal(err)
	}
	return models
}

// everySurface is every surface core defines, in the order derivedSurfaces
// reports them.
func everySurface() []core.ModelSurface {
	return []core.ModelSurface{
		core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses, core.ModelSurfaceMessages,
		core.ModelSurfaceEmbeddings, core.ModelSurfaceAudioTranscriptions, core.ModelSurfaceAudioSpeech,
		core.ModelSurfaceImages, core.ModelSurfaceVideos,
	}
}

// derivedSurfaces are the surfaces core infers a row serves, in the order of
// everySurface: each chat surface its inferred capabilities support, and
// each other surface whose operation they support.
func derivedSurfaces(row core.ModelInfo) []core.ModelSurface {
	capabilities := core.InferCapabilities(row, time.Time{}, time.Time{})
	operations := map[core.ModelSurface]core.ModelOperation{
		core.ModelSurfaceEmbeddings:          core.ModelOperationEmbeddings,
		core.ModelSurfaceAudioTranscriptions: core.ModelOperationAudioIn,
		core.ModelSurfaceAudioSpeech:         core.ModelOperationAudioOut,
		core.ModelSurfaceImages:              core.ModelOperationImage,
		core.ModelSurfaceVideos:              core.ModelOperationVideo,
	}
	var surfaces []core.ModelSurface
	for _, surface := range everySurface() {
		support := capabilities.SurfaceCompatibility(surface)
		if operation, ok := operations[surface]; ok {
			support = capabilities.OperationCompatibility(operation)
		}
		if support.IsSupported() {
			surfaces = append(surfaces, surface)
		}
	}
	return surfaces
}

func TestSpeechProviderListsItsModelsAsSpeechModels(t *testing.T) {
	provider := catalogProvider(t, []core.ModelSurface{core.ModelSurfaceAudioSpeech},
		[]core.ModelInfo{{ID: "voice-1", Object: "model"}, {ID: "voice-2", Object: "model"}})
	models := listModels(t, provider)
	if len(models) != 2 {
		t.Fatalf("models = %+v", models)
	}
	for _, model := range models {
		if !slices.Equal(model.SupportedAPIs, []string{"/v1/audio/speech"}) {
			t.Fatalf("%s SupportedAPIs = %q", model.ID, model.SupportedAPIs)
		}
		if got := derivedSurfaces(model); !slices.Equal(got, []core.ModelSurface{core.ModelSurfaceAudioSpeech}) {
			t.Fatalf("%s derived surfaces = %v", model.ID, got)
		}
		inferred := core.InferCapabilities(model, time.Time{}, time.Time{})
		if inferred.Operations.AudioOut != core.SupportSupported || inferred.Operations.Chat != core.SupportUnknown ||
			inferred.Surfaces != (core.ModelSurfaceCapabilities{}) {
			t.Fatalf("%s operations %+v, surfaces %+v", model.ID, inferred.Operations, inferred.Surfaces)
		}
		if interfaces := core.TransportInterfaces(provider, nil, model.ID, model.SupportedAPIs); len(interfaces) != 0 {
			t.Fatalf("%s offers chat interfaces %+v", model.ID, interfaces)
		}
		if native, emulated, unknown := core.ListingSurfaces(provider, nil, core.TransportEvidence{Row: model}, time.Now()); native != nil || emulated != nil || unknown != nil {
			t.Fatalf("%s lists chat surfaces: native %v, emulated %v, unknown %v", model.ID, native, emulated, unknown)
		}
	}

	// Each row has its own paths, and the provider's surfaces stay its own.
	models[0].SupportedAPIs[0] = "/v1/changed"
	if models[1].SupportedAPIs[0] != "/v1/audio/speech" || listModels(t, provider)[0].SupportedAPIs[0] != "/v1/audio/speech" ||
		!slices.Equal(provider.Info().Surfaces, []core.ModelSurface{core.ModelSurfaceAudioSpeech}) {
		t.Fatal("changing one row's SupportedAPIs changed another row or the provider")
	}
}

func TestChatProviderListsItsModelsWithChatCompletions(t *testing.T) {
	provider := catalogProvider(t, []core.ModelSurface{core.ModelSurfaceChatCompletions}, []core.ModelInfo{{ID: "chat-1", Object: "model"}})
	models := listModels(t, provider)
	if len(models) != 1 || !slices.Equal(models[0].SupportedAPIs, []string{"/v1/chat/completions"}) {
		t.Fatalf("models = %+v", models)
	}
	model := models[0]
	if got := derivedSurfaces(model); !slices.Equal(got, []core.ModelSurface{core.ModelSurfaceChatCompletions}) {
		t.Fatalf("derived surfaces = %v", got)
	}
	if surface := core.ParseSurfacePath(model.SupportedAPIs[0]); surface != core.ModelSurfaceChatCompletions {
		t.Fatalf("ParseSurfacePath(%q) = %q", model.SupportedAPIs[0], surface)
	}
	if inferred := core.InferCapabilities(model, time.Time{}, time.Time{}); inferred.Operations.Chat != core.SupportSupported {
		t.Fatalf("operations = %+v", inferred.Operations)
	}
	if interfaces := core.TransportInterfaces(provider, nil, model.ID, model.SupportedAPIs); len(interfaces) != 1 ||
		interfaces[0].Surface != core.ModelSurfaceChatCompletions {
		t.Fatalf("interfaces = %+v", interfaces)
	}
}

// A row that says anything about what its model serves, even that it does
// not serve something, is kept as the daemon sent it. Capabilities that name
// no operation or surface, such as limits or features, say nothing about it.
func TestModelsThatReportWhatTheyServeKeepIt(t *testing.T) {
	contextTokens := int64(8192)
	rows := []core.ModelInfo{
		{ID: "own-apis", Object: "model", SupportedAPIs: []string{"/responses"}},
		{ID: "unrecognized-apis", Object: "model", SupportedAPIs: []string{"/v1/rerank"}},
		{ID: "typed", Object: "model", Capabilities: &core.ModelCapabilities{
			SchemaVersion: core.ModelCapabilitiesSchemaVersion,
			Operations:    core.ModelOperationCapabilities{AudioOut: core.SupportSupported},
		}},
		{ID: "legacy", Object: "model", LegacyCapabilities: map[string]any{"tts": true}},
		{ID: "legacy-refusal", Object: "model", LegacyCapabilities: map[string]any{"chat": false}},
		{ID: "silent", Object: "model"},
		{ID: "features-only", Object: "model",
			LegacyCapabilities: map[string]any{"vision": true, "context_window": float64(8192)},
			Capabilities: &core.ModelCapabilities{
				SchemaVersion: core.ModelCapabilitiesSchemaVersion,
				Limits:        core.ModelCapabilityLimits{ContextTokens: &contextTokens},
			}},
	}
	provider := catalogProvider(t, []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceAudioSpeech}, rows)
	models := listModels(t, provider)
	if len(models) != len(rows) {
		t.Fatalf("models = %+v", models)
	}
	for index, model := range models {
		want := rows[index]
		if want.ID == "silent" || want.ID == "features-only" {
			want.SupportedAPIs = []string{"/v1/chat/completions", "/v1/audio/speech"}
		}
		if !reflect.DeepEqual(model, want) {
			t.Errorf("model %s = %+v\nwant %+v", want.ID, model, want)
		}
	}
}

// Core infers from a filled row exactly the provider's surfaces, those core
// defines, and reads its chat surfaces in the provider's order.
func TestCoreDerivesTheProviderSurfacesFromAFilledModel(t *testing.T) {
	for name, surfaces := range map[string][]core.ModelSurface{
		"speech":             {core.ModelSurfaceAudioSpeech},
		"chat":               {core.ModelSurfaceChatCompletions},
		"responses and chat": {core.ModelSurfaceResponses, core.ModelSurfaceChatCompletions},
		"every surface":      everySurface(),
		"listed twice":       {core.ModelSurfaceMessages, core.ModelSurfaceImages, core.ModelSurfaceMessages},
		"undefined":          {"telepathy", core.ModelSurfaceEmbeddings},
	} {
		t.Run(name, func(t *testing.T) {
			model := listModels(t, catalogProvider(t, surfaces, []core.ModelInfo{{ID: "model", Object: "model"}}))[0]
			var want []core.ModelSurface
			for _, surface := range everySurface() {
				if slices.Contains(surfaces, surface) {
					want = append(want, surface)
				}
			}
			if got := derivedSurfaces(model); !slices.Equal(got, want) {
				t.Fatalf("derived surfaces = %v from %q, want %v", got, model.SupportedAPIs, want)
			}
			var wantChat, gotChat []core.ModelSurface
			for _, surface := range surfaces {
				if core.ParseSurfacePath(core.SurfacePath(surface)) != "" && !slices.Contains(wantChat, surface) {
					wantChat = append(wantChat, surface)
				}
			}
			for _, path := range model.SupportedAPIs {
				if surface := core.ParseSurfacePath(path); surface != "" {
					gotChat = append(gotChat, surface)
				}
			}
			if !slices.Equal(gotChat, wantChat) || len(model.SupportedAPIs) != len(want) {
				t.Fatalf("SupportedAPIs = %q, chat surfaces %v, want %v", model.SupportedAPIs, gotChat, wantChat)
			}
		})
	}
}

func TestProviderListModelsFailsAsTheClientDoes(t *testing.T) {
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		writeJSON(t, w, http.StatusServiceUnavailable, extension.ErrorResponse{Error: extension.ErrorDetail{Message: "catalog unavailable", Code: 503}})
	})
	provider := extension.NewProvider(client, extension.ProviderInfo{ID: "catalog", Surfaces: []core.ModelSurface{core.ModelSurfaceAudioSpeech}})
	models, err := provider.ListModels(context.Background(), nil)
	_, clientErr := client.ListModels(context.Background(), "catalog", nil)
	var providerError *core.ProviderError
	if models != nil || !errors.As(err, &providerError) || clientErr == nil || err.Error() != clientErr.Error() ||
		core.ClassifyError(err) != core.ClassifyError(clientErr) || !strings.Contains(err.Error(), "catalog unavailable") {
		t.Fatalf("models = %+v, error = %v, the client's = %v", models, err, clientErr)
	}
}

func TestProviderIDsAreOneSafePathSegment(t *testing.T) {
	client, calls := daemon(t, func(w http.ResponseWriter, r *http.Request) {})
	for _, id := range []string{"", "a/b", ".", "..", "info", "a b", "p?x=1", "ä", strings.Repeat("a", 129)} {
		_, err := client.Invoke(context.Background(), id, core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "m"})
		var providerError *core.ProviderError
		if !errors.As(err, &providerError) || providerError.Class != core.ProviderErrorConfiguration {
			t.Errorf("provider id %q error = %v", id, err)
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("invalid provider ids reached the daemon %d times", calls.Load())
	}
}

func TestLossesHeaderIgnoresGarbage(t *testing.T) {
	header := http.Header{}
	extension.SetLossesHeader(header, nil)
	if header.Get(extension.HeaderLosses) != "" {
		t.Fatal("no losses set a header")
	}
	header.Set(extension.HeaderLosses, "%%%")
	if extension.LossesFromHeader(header) != nil {
		t.Fatal("undecodable losses decoded")
	}
	header.Set(extension.HeaderLosses, base64.StdEncoding.EncodeToString([]byte("[not json")))
	if extension.LossesFromHeader(header) != nil {
		t.Fatal("invalid losses JSON decoded")
	}
}
