package providers

import (
	"context"
	"reflect"
	"sync"
	"testing"
	"time"

	copilotauth "github.com/xibodev/llm-provider-auth/copilot"
	core "github.com/xibodev/llmgw-core"
)

func invokeCopilotChat(t *testing.T, provider *Copilot, credential *core.Credential) error {
	t.Helper()
	_, err := provider.Invoke(context.Background(), copilotRequest(core.ModelSurfaceChatCompletions, "gpt-fixture", `{"messages":[]}`, credential))
	return err
}

func sessionsUsed(calls []copilotRecorded) []string {
	used := make([]string, len(calls))
	for index, call := range calls {
		used[index] = call.authorization
	}
	return used
}

func TestCopilotKeepsOneSessionPerCredential(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilotChat)
	provider := newFixtureCopilot(t, backend)
	for _, token := range []string{"oauth-a", "oauth-a", "oauth-b", " oauth-a ", "oauth-b"} {
		if err := invokeCopilotChat(t, provider, copilotCredential(token)); err != nil {
			t.Fatal(err)
		}
	}
	exchanges, calls := backend.take()
	if !reflect.DeepEqual(exchanges, []string{"oauth-a", "oauth-b"}) {
		t.Fatalf("exchanges = %v, want one per credential", exchanges)
	}
	want := []string{"Bearer session-1", "Bearer session-1", "Bearer session-2", "Bearer session-1", "Bearer session-2"}
	if used := sessionsUsed(calls); !reflect.DeepEqual(used, want) {
		t.Fatalf("sessions = %v, want %v", used, want)
	}
}

// Concurrent requests for one credential share its one exchange.
func TestCopilotSharesAnExchangeBetweenConcurrentRequests(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilotChat)
	provider := newFixtureCopilot(t, backend)
	var group sync.WaitGroup
	errs := make(chan error, 16)
	for range 16 {
		group.Go(func() { errs <- invokeCopilotChat(t, provider, copilotCredential("oauth-shared")) })
	}
	group.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if exchanges, calls := backend.take(); len(exchanges) != 1 || len(calls) != 16 {
		t.Fatalf("exchanges = %v, calls = %d, want one exchange for 16 calls", exchanges, len(calls))
	}
}

func TestCopilotReplacesASessionNearItsExpiry(t *testing.T) {
	t.Parallel()
	backend := newCopilotBackend(t, answerCopilotChat)
	backend.update(func(b *copilotBackend) { b.expiresAt = time.Now().Add(30 * time.Second).Unix() })
	provider := newFixtureCopilot(t, backend)
	for range 2 {
		if err := invokeCopilotChat(t, provider, copilotCredential("oauth-a")); err != nil {
			t.Fatal(err)
		}
	}
	if exchanges, _ := backend.take(); len(exchanges) != 2 {
		t.Fatalf("exchanges = %v, want a new session for each request", exchanges)
	}
}

// Without a credential the product's own session is Auth's to cache, as
// in the gateway: on disk with a cache directory, not at all without.
func TestCopilotLeavesTheProductSessionToAuth(t *testing.T) {
	t.Parallel()
	for name, cacheDir := range map[string]string{"no cache directory": "", "cache directory": t.TempDir()} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			backend := newCopilotBackend(t, answerCopilotChat)
			provider := newFixtureCopilot(t, backend, func(_ *CopilotConfig, auth *copilotauth.Config) { auth.CacheDir = cacheDir })
			for range 2 {
				if err := invokeCopilotChat(t, provider, nil); err != nil {
					t.Fatal(err)
				}
			}
			want := []string{"product-oauth", "product-oauth"}
			if cacheDir != "" {
				want = want[:1]
			}
			if exchanges, _ := backend.take(); !reflect.DeepEqual(exchanges, want) {
				t.Fatalf("exchanges = %v, want %v", exchanges, want)
			}
		})
	}
}

// Copilot rejecting a session is retried once with a session exchanged
// afresh, past Auth's disk cache, which still holds the rejected one.
func TestCopilotReplaysOnceWithANewSessionWhenCopilotRejectsOne(t *testing.T) {
	t.Parallel()
	for name, credential := range map[string]*core.Credential{"credential": copilotCredential("oauth-a"), "product": nil} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			backend := newCopilotBackend(t, answerCopilotChat)
			backend.update(func(b *copilotBackend) { b.rejected["session-1"] = true })
			provider := newFixtureCopilot(t, backend, func(_ *CopilotConfig, auth *copilotauth.Config) { auth.CacheDir = t.TempDir() })
			for range 2 {
				if err := invokeCopilotChat(t, provider, credential); err != nil {
					t.Fatal(err)
				}
			}
			exchanges, calls := backend.take()
			if len(exchanges) != 2 {
				t.Fatalf("exchanges = %v, want the rejected session replaced once", exchanges)
			}
			want := []string{"Bearer session-1", "Bearer session-2", "Bearer session-2"}
			if used := sessionsUsed(calls); !reflect.DeepEqual(used, want) {
				t.Fatalf("sessions = %v, want %v", used, want)
			}
		})
	}
}
