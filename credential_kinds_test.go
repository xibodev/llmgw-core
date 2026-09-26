package core_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/xibodev/llm-provider-auth/gcp"
	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
)

func TestCredentialKindsCarryTheirMaterialInToken(t *testing.T) {
	for kind, material := range map[string]string{
		core.TokenTypeGCPServiceAccount: `{"type":"service_account","project_id":"fixture-project"}`,
	} {
		t.Run(kind, func(t *testing.T) {
			credential := core.CredentialFromRecord("fixture-key", tokenstore.Record{
				AccessToken: material, TokenType: kind,
				Metadata: map[string]string{core.CredentialMetadataProjectID: "fixture-billing"},
			})
			if credential.Token != material || credential.APIKey != "" || credential.TokenType != kind ||
				credential.Metadata[core.CredentialMetadataProjectID] != "fixture-billing" {
				t.Fatalf("credential = %v, token set: %v", credential, credential.Token == material)
			}
			rendered := fmt.Sprint(credential) + fmt.Sprintf("%#v %+v", *credential, *credential)
			if strings.Contains(rendered, material) || strings.Contains(rendered, "fixture-billing") {
				t.Fatalf("the credential printed its material: %s", rendered)
			}
		})
	}
}

// One spelling, so a record llm-provider-auth's gcp package names and a
// credential core names are the same kind.
func TestGCPServiceAccountKindIsLLMProviderAuths(t *testing.T) {
	if core.TokenTypeGCPServiceAccount != gcp.CredentialKind {
		t.Fatalf("TokenTypeGCPServiceAccount = %q, llm-provider-auth names the kind %q", core.TokenTypeGCPServiceAccount, gcp.CredentialKind)
	}
}
