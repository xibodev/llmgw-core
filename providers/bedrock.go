package providers

import (
	"fmt"
	"strings"

	core "github.com/xibodev/llmgw-core"
)

// BedrockDefaultRegion is the region NewBedrock serves when given none, as
// the gateway does.
const BedrockDefaultRegion = "us-east-1"

// NewBedrock returns an OpenAICompatible provider for Amazon Bedrock's
// OpenAI-compatible endpoint, which replaces config.BaseURL. baseURL, when
// set, is the endpoint, such as a bedrock-mantle or VPC endpoint; otherwise
// it is https://bedrock-runtime.<region>.amazonaws.com/v1, in
// BedrockDefaultRegion when region is blank.
//
// A Bedrock API key is a bearer token, so a credential authenticates as it
// does for any OpenAI-compatible upstream: nothing is signed and no AWS SDK
// is needed. A region that is not a region name is refused before any
// request, so a key never reaches a host the region did not name.
func NewBedrock(region, baseURL string, config OpenAICompatibleConfig) (*OpenAICompatible, error) {
	config.BaseURL = strings.TrimSpace(baseURL)
	if config.BaseURL == "" {
		region = strings.ToLower(strings.TrimSpace(region))
		if region == "" {
			region = BedrockDefaultRegion
		}
		if !bedrockRegion(region) {
			return nil, core.NewConfigurationError(fmt.Sprintf("%q is not an AWS region name", region), nil)
		}
		config.BaseURL = "https://bedrock-runtime." + region + ".amazonaws.com/v1"
	}
	provider, err := NewOpenAICompatible(config)
	if err != nil {
		return nil, err
	}
	provider.label = "Bedrock"
	return provider, nil
}

// bedrockRegion reports a region name: groups of lowercase letters and
// digits joined by single hyphens, such as us-gov-west-1.
func bedrockRegion(region string) bool {
	for _, group := range strings.Split(region, "-") {
		if group == "" {
			return false
		}
		for _, character := range group {
			if (character < 'a' || character > 'z') && (character < '0' || character > '9') {
				return false
			}
		}
	}
	return true
}
