package core_test

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"testing"

	core "github.com/xibodev/llmgw-core"
)

// fixtureCredential sets every field, with synthetic secrets.
func fixtureCredential() core.Credential {
	return core.Credential{
		APIKey: "fixture-api-key-value", Token: "fixture-token-value",
		ConnectionID: "user-1-antigravity", CredentialRevision: 7,
		Headers:   map[string]string{"X-Fixture": "fixture-header-value"},
		AccountID: "account-1", TokenType: "Bearer",
		Metadata: map[string]string{
			"project_id": "fixture-project-value", "client_secret": "fixture-client-secret-value", "client_id": "fixture-client-id-value",
		},
	}
}

// leakedSecret returns a secret of credential that rendered reveals, as is or
// hex-encoded the way %x and %X print it.
func leakedSecret(rendered string, credential core.Credential) (string, bool) {
	secrets := []string{credential.APIKey, credential.Token}
	for _, values := range []map[string]string{credential.Headers, credential.Metadata} {
		for _, value := range values {
			secrets = append(secrets, value)
		}
	}
	for _, secret := range secrets {
		if strings.Contains(rendered, secret) || strings.Contains(strings.ToLower(rendered), hex.EncodeToString([]byte(secret))) {
			return secret, true
		}
	}
	return "", false
}

func TestCredentialNeverPrintsSecrets(t *testing.T) {
	t.Parallel()
	credential := fixtureCredential()
	const want = `core.Credential{ConnectionID:"user-1-antigravity" CredentialRevision:7 AccountID:"account-1" TokenType:"Bearer" ` +
		`APIKey:redacted Token:redacted Headers:redacted Metadata:map[client_id:redacted client_secret:redacted project_id:redacted]}`
	for _, rendered := range []string{
		credential.String(), credential.GoString(), fmt.Sprint(credential), fmt.Sprint(&credential),
		fmt.Sprintf("%+v", credential), fmt.Sprintf("%#v", &credential), fmt.Sprintf("%s", credential),
	} {
		if rendered != want {
			t.Fatalf("rendered %s, want %s", rendered, want)
		}
	}
	const empty = `core.Credential{ConnectionID:"" CredentialRevision:0 AccountID:"" TokenType:"" APIKey:absent Token:absent Headers:absent Metadata:map[]}`
	if rendered := fmt.Sprint(core.Credential{}); rendered != empty {
		t.Fatalf("empty credential rendered %s, want %s", rendered, empty)
	}
	if rendered := fmt.Sprint((*core.Credential)(nil)); rendered != "<nil>" {
		t.Fatalf("nil credential rendered %s", rendered)
	}

	// Every verb fmt hands to a credential, alone or inside a value that
	// carries one. %p of a value and %w are bad verbs, which fmt prints
	// without calling any method, so no type can guard them.
	holders := []any{
		credential, &credential, core.Request{Credential: &credential},
		[]core.Credential{credential}, map[string]core.Credential{"openai": credential},
	}
	for _, verb := range []string{
		"%v", "%+v", "%#v", "%s", "%q", "%x", "%X", "%T", "%t", "%b", "%c", "%d",
		"%o", "%O", "%U", "%e", "%E", "%f", "%F", "%g", "%G", "%-40.9v",
	} {
		for _, holder := range holders {
			rendered := fmt.Sprintf(verb, holder)
			if secret, leaked := leakedSecret(rendered, credential); leaked {
				t.Fatalf("%s of %T printed %q: %s", verb, holder, secret, rendered)
			}
		}
	}
}

func TestCredentialNeverLogsSecrets(t *testing.T) {
	t.Parallel()
	credential := fixtureCredential()
	cases := map[string]struct {
		open func(io.Writer) slog.Handler
		want string
	}{
		"json": {
			open: func(w io.Writer) slog.Handler { return slog.NewJSONHandler(w, nil) },
			want: `"credential":{"connection_id":"user-1-antigravity","credential_revision":7,"account_id":"account-1","token_type":"Bearer",` +
				`"has_api_key":true,"has_token":true,"has_headers":true,"metadata_keys":["client_id","client_secret","project_id"]}`,
		},
		"text": {
			open: func(w io.Writer) slog.Handler { return slog.NewTextHandler(w, nil) },
			want: `credential.connection_id=user-1-antigravity credential.credential_revision=7 credential.account_id=account-1 ` +
				`credential.token_type=Bearer credential.has_api_key=true credential.has_token=true credential.has_headers=true ` +
				`credential.metadata_keys="[client_id client_secret project_id]"`,
		},
	}
	for name, tc := range cases {
		var logged bytes.Buffer
		logger := slog.New(tc.open(&logged))
		logger.Info("value", "credential", credential)
		logger.Info("pointer", "credential", &credential)
		output := logged.String()
		if secret, leaked := leakedSecret(output, credential); leaked {
			t.Fatalf("%s handler logged %q: %s", name, secret, output)
		}
		if strings.Count(output, tc.want) != 2 {
			t.Fatalf("%s handler logged %s, want both lines to hold %s", name, output, tc.want)
		}
	}
}

func TestCredentialMetadataNeverSerializes(t *testing.T) {
	t.Parallel()
	credential := fixtureCredential()
	encoded, err := json.Marshal(credential)
	if err != nil {
		t.Fatal(err)
	}
	for key, value := range credential.Metadata {
		if bytes.Contains(encoded, []byte(key)) || bytes.Contains(encoded, []byte(value)) {
			t.Fatalf("metadata %s serialized: %s", key, encoded)
		}
	}
	if !bytes.Contains(encoded, []byte(`"account_id":"account-1","token_type":"Bearer"`)) {
		t.Fatalf("account and token type did not serialize: %s", encoded)
	}
}
