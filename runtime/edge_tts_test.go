package runtime_test

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/providers"
	coreruntime "github.com/xibodev/llmgw-core/runtime"
	"github.com/xibodev/llmgw-core/translation"
)

// The frames the gateway sends to speak "Say hello" with en-US-TestNeural at
// the fixture's clock, IDs and token, as its own functions produce them.
const (
	edgeTTSRuntimeConfig = "X-Timestamp:Wed Mar 04 2026 05:06:07 GMT+0000 (Coordinated Universal Time)\r\n" +
		"Content-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n" +
		`{"context":{"synthesis":{"audio":{"metadataoptions":{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"true"},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}`
	edgeTTSRuntimeSSML = "X-RequestId:fedcba9876543210fedcba9876543210\r\nContent-Type:application/ssml+xml\r\n" +
		"X-Timestamp:Wed Mar 04 2026 05:06:07 GMT+0000 (Coordinated Universal Time)Z\r\nPath:ssml\r\n\r\n" +
		"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='en-US-TestNeural'>" +
		"<prosody pitch='+0Hz' rate='+0%' volume='+0%'>Say hello</prosody></voice></speak>"
	edgeTTSRuntimeQuery = "?Ocp-Apim-Subscription-Key=fixture-token&Sec-MS-GEC=ADB9188F52EF5B80F8AD80F06FA7A17D58DE9EE72945BD4FE3C329599B5CBD02" +
		"&Sec-MS-GEC-Version=1-140.0.3485.14&ConnectionId=0123456789abcdef0123456789abcdef"
)

// edgeTTSSocket is an in-memory synthesis websocket: it records what Edge
// TTS writes and answers the SSML with one audio message and turn.end.
type edgeTTSSocket struct {
	mu      sync.Mutex
	url     string
	written []string
	replies []string
}

func (s *edgeTTSSocket) dial(_ context.Context, url string, _ http.Header, _ []string) (providers.WebSocketConn, *http.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.url, s.written = url, nil
	headers := "Path:audio\r\n"
	audio := make([]byte, 2, 2+len(headers)+len("MP3-FIXTURE"))
	binary.BigEndian.PutUint16(audio, uint16(len(headers)))
	s.replies = []string{string(append(append(audio, headers...), "MP3-FIXTURE"...)), "Path:turn.end\r\n\r\n{}"}
	return s, nil, nil
}

func (s *edgeTTSSocket) WriteText(_ context.Context, data []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.written = append(s.written, string(data))
	return nil
}

func (s *edgeTTSSocket) Read(context.Context) (int, []byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.written) < 2 || len(s.replies) == 0 {
		return 0, nil, io.ErrUnexpectedEOF
	}
	reply := s.replies[0]
	s.replies = s.replies[1:]
	if strings.HasPrefix(reply, "Path:") {
		return providers.WebSocketTextMessage, []byte(reply), nil
	}
	return providers.WebSocketBinaryMessage, []byte(reply), nil
}

func (s *edgeTTSSocket) Close() error { return nil }

func TestRuntimeServesTheEdgeTTSVertical(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	voices := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tts/voices/list" || r.URL.Query().Get("Ocp-Apim-Subscription-Key") != "fixture-token" {
			http.NotFound(w, r)
			return
		}
		_, _ = io.WriteString(w, `[{"ShortName":"en-US-TestNeural","FriendlyName":"Test voice","Locale":"en-US","Gender":"Female"}]`)
	}))
	defer voices.Close()
	socket := &edgeTTSSocket{}
	runtime, err := coreruntime.New(coreruntime.Options[string]{
		Settings: coreruntime.NewMemorySettings(voices.URL + "/tts"),
		Providers: func(base string, _ string) (core.Provider, error) {
			ids := []string{"0123456789abcdef0123456789abcdef", "fedcba9876543210fedcba9876543210"}
			var mu sync.Mutex
			provider, err := providers.NewEdgeTTS(providers.EdgeTTSConfig{
				Dial: socket.dial, BaseURL: base, Client: voices.Client(), DefaultVoice: "en-US-TestNeural",
				Now: func() time.Time { return time.Date(2026, time.March, 4, 5, 6, 7, 0, time.UTC) },
				NewID: func() string {
					mu.Lock()
					defer mu.Unlock()
					id := ids[0]
					ids = append(ids[1:], id)
					return id
				},
			})
			if err != nil {
				return nil, err
			}
			return translation.Adapter{Provider: provider}, nil
		},
		Credentials: edgeTTSCredentials(t),
	})
	if err != nil {
		t.Fatal(err)
	}
	owner := core.Caller{ID: "owner", Kind: core.CallerHuman}
	speech := core.Request{
		Surface: core.ModelSurfaceAudioSpeech, Model: "default", ContentType: core.ContentTypeJSON,
		Body: []byte(`{"model":"tts-1","input":"Say hello","voice":"en-US-TestNeural"}`),
	}

	t.Run("speech", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, owner, "edge_tts", speech)
		if err != nil || string(response.Body) != "MP3-FIXTURE" || response.ContentType != "audio/mpeg" || len(response.Losses) != 0 {
			t.Fatalf("response = %q %q %v, err = %v", response.Body, response.ContentType, response.Losses, err)
		}
		socket.mu.Lock()
		defer socket.mu.Unlock()
		if want := "ws://" + strings.TrimPrefix(voices.URL, "http://") + "/tts/websocket/v1" + edgeTTSRuntimeQuery; socket.url != want {
			t.Fatalf("dialed %s\nwant   %s", socket.url, want)
		}
		if !reflect.DeepEqual(socket.written, []string{edgeTTSRuntimeConfig, edgeTTSRuntimeSSML}) {
			t.Fatalf("frames = %q", socket.written)
		}
	})

	t.Run("speech does not stream, and chat is no surface of it", func(t *testing.T) {
		if _, err := runtime.Stream(ctx, owner, "edge_tts", speech); core.ClassifyError(err).Disposition() != core.DispositionFailover {
			t.Fatalf("stream: err = %v, want a refusal that permits failover", err)
		}
		chat := core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "default", ContentType: core.ContentTypeJSON, Body: []byte(`{"messages":[]}`)}
		var surfaceErr *core.SurfaceError
		if _, err := runtime.Invoke(ctx, owner, "edge_tts", chat); !errors.As(err, &surfaceErr) {
			t.Fatalf("chat: err = %v, want a surface error", err)
		}
	})

	t.Run("catalog", func(t *testing.T) {
		record, err := runtime.ListModels(ctx, owner, "edge_tts")
		if err != nil || len(record.Evidence.Models) != 1 || record.Evidence.Models[0].ID != "en-US-TestNeural" || record.Evidence.Models[0].DisplayName != "Test voice" {
			t.Fatalf("catalog = %+v, err = %v", record, err)
		}
	})
}

// edgeTTSCredentials binds the owner to the fixture access token.
func edgeTTSCredentials(t *testing.T) core.CredentialStore {
	t.Helper()
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(context.Background(), "owner-edge-tts", core.APIKeyRecord("fixture-token")); err != nil {
		t.Fatal(err)
	}
	store.Bind(core.Caller{ID: "owner", Kind: core.CallerHuman}, "edge_tts", "owner-edge-tts")
	return store
}
