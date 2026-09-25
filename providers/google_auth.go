package providers

import (
	"fmt"
	"net/http"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// googleAccess is the credential material of one operation and, on Vertex
// AI, the project the operation is billed to.
type googleAccess struct {
	apiKey  string
	bearer  string
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
// Vertex AI refuses a request that presents both. A credential of a kind
// Google does not serve is refused, so its material is never sent as a
// bearer.
//
// Vertex AI resolves the project too, and needs a credential: the gateway
// fails without one rather than send a request Vertex AI answers with a
// 401, which reads as a rejected key instead of an absent one.
func (p *Google) access(credential *core.Credential) (googleAccess, error) {
	var access googleAccess
	credentialProject := ""
	if credential != nil {
		credentialProject = strings.TrimSpace(credential.Metadata[core.CredentialMetadataProjectID])
		access.apiKey = strings.TrimSpace(credential.APIKey)
		switch kind := strings.TrimSpace(credential.TokenType); {
		case strings.EqualFold(kind, core.TokenTypeAPIKey):
			if access.apiKey == "" {
				access.apiKey = strings.TrimSpace(credential.Token)
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
	if access.apiKey == "" && access.bearer == "" {
		return googleAccess{}, core.NewConfigurationError("vertex_ai: no credential configured — vertex_ai needs an API key, an access token "+
			"or a Google service account key; add one before sending requests", nil)
	}
	project, err := p.vertexProject(credentialProject, "credential")
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

// authorization is the header an operation authenticates with.
func (p *Google) authorization(access googleAccess) (googleAuthorization, error) {
	switch {
	case access.bearer != "":
		return googleAuthorization{name: "Authorization", value: "Bearer " + access.bearer}, nil
	case access.apiKey != "":
		return googleAuthorization{name: "x-goog-api-key", value: access.apiKey}, nil
	}
	return googleAuthorization{}, nil
}
