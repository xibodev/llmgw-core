package oauthflow_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	core "github.com/xibodev/llmgw-core"
	"github.com/xibodev/llmgw-core/oauthflow"
)

const fixtureInstance = "fixture-provider"

type testClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *testClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *testClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

type harness struct {
	service     *oauthflow.Service
	credentials *core.MemoryCredentialStore
	clock       *testClock
	driver      *fakeDriver
}

func newHarness(t *testing.T) *harness {
	t.Helper()
	h := &harness{
		credentials: core.NewMemoryCredentialStore(),
		clock:       &testClock{now: time.Unix(1_800_000_000, 0).UTC()},
		driver:      &fakeDriver{},
	}
	service, err := oauthflow.New(oauthflow.Options{
		Store:       oauthflow.NewMemoryFlowStore(oauthflow.MemoryFlowStoreOptions{Now: h.clock.Now}),
		Credentials: h.credentials,
		Drivers: func(instance string, _ oauthflow.Method) (oauthflow.Driver, error) {
			if instance != fixtureInstance {
				return nil, fmt.Errorf("instance %q offers no OAuth", instance)
			}
			return h.driver, nil
		},
		CredentialKey: func(_ context.Context, completion oauthflow.Completion) (string, error) {
			name := completion.Params["connection_name"]
			if name == "" {
				name = "default"
			}
			return completion.Caller.ID + "/" + completion.Instance + "/" + name, nil
		},
		Now: h.clock.Now,
	})
	if err != nil {
		t.Fatal(err)
	}
	h.service = service
	return h
}

func (h *harness) start(t *testing.T, method oauthflow.Method, options ...oauthflow.StartOption) oauthflow.View {
	t.Helper()
	view, err := h.service.Start(context.Background(), owner(), fixtureInstance, method, options...)
	if err != nil {
		t.Fatalf("Start %s: %v", method, err)
	}
	assertNoSecrets(t, view)
	return view
}

// stateOf reads the state out of a view's authorization URL, as a provider
// would echo it back.
func stateOf(t *testing.T, view oauthflow.View) string {
	t.Helper()
	_, state, found := strings.Cut(view.AuthorizationURL, "state=")
	if !found || state == "" {
		t.Fatalf("authorization URL %q carries no state", view.AuthorizationURL)
	}
	return state
}

func owner() core.Caller {
	return core.Caller{ID: "user-1", Kind: core.CallerHuman, ProjectID: "project-1"}
}

// intruders share one or two identity fields with owner, never all three.
func intruders() map[string]core.Caller {
	return map[string]core.Caller{
		"another user":        {ID: "user-2", Kind: core.CallerHuman, ProjectID: "project-1"},
		"same id, other kind": {ID: "user-1", Kind: core.CallerService, ProjectID: "project-1"},
		"same id, other proj": {ID: "user-1", Kind: core.CallerHuman, ProjectID: "project-2"},
		"anonymous":           {Kind: core.CallerAnonymous},
	}
}

// assertNoSecrets checks a view's JSON and its %v, %+v and %#v formatting.
// The state may appear only inside AuthorizationURL, so that is removed
// first, in both its plain and its JSON-escaped form.
func assertNoSecrets(t *testing.T, view oauthflow.View) {
	t.Helper()
	text := visibleText(view)
	for _, secret := range allSecrets() {
		if strings.Contains(text, secret) {
			t.Fatalf("view exposes %q: %s", secret, text)
		}
	}
	// The authorization URL names redirect_uri itself; only fields count.
	fields := view
	fields.AuthorizationURL = ""
	encoded, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"verifier", "device_code", "redirect", "driver", "params", "token"} {
		if strings.Contains(strings.ToLower(string(encoded)), field) {
			t.Fatalf("view JSON has a %q field: %s", field, encoded)
		}
	}
}

// visibleText is a view's JSON and its %v, %+v and %#v formatting, with its
// AuthorizationURL removed in both plain and JSON-escaped form.
func visibleText(view oauthflow.View) string {
	encoded, _ := json.Marshal(view)
	text := string(encoded) + fmt.Sprintf(" %v %+v %#v", view, view, view)
	if view.AuthorizationURL != "" {
		escapedURL, _ := json.Marshal(view.AuthorizationURL)
		text = strings.ReplaceAll(text, strings.Trim(string(escapedURL), `"`), "")
		text = strings.ReplaceAll(text, view.AuthorizationURL, "")
	}
	return text
}

func wantErr(t *testing.T, err, want error, what string) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Fatalf("%s: err=%v, want %v", what, err, want)
	}
}
