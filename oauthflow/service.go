package oauthflow

import (
	"context"
	"errors"
	"maps"
	"strings"
	"time"

	"github.com/xibodev/llm-provider-auth/tokenstore"

	core "github.com/xibodev/llmgw-core"
)

var (
	// ErrSlowDown reports a device poll that came before the interval
	// elapsed, or a provider's slow_down. The flow is still pending; poll
	// again after View.Interval seconds.
	ErrSlowDown = errors.New("oauthflow: poll again after the interval")
	// ErrAccessDenied reports that the owner or the provider refused the
	// authorization. The flow has ended.
	ErrAccessDenied = errors.New("oauthflow: authorization was denied")
	// ErrStateMismatch reports a completion whose OAuth state is not the
	// flow's. The flow has ended.
	ErrStateMismatch = errors.New("oauthflow: OAuth state does not match the flow")
	// ErrWrongMethod reports an operation the flow's method does not offer,
	// such as polling a browser flow.
	ErrWrongMethod = errors.New("oauthflow: the flow's method does not offer this operation")
	// ErrCodeRequired reports a completion that carries neither a code nor
	// a provider error. The flow is untouched.
	ErrCodeRequired = errors.New("oauthflow: authorization code is required")
)

// DefaultTTL is how long a flow lives when its driver names no expiry.
const DefaultTTL = 10 * time.Minute

// slowDownStep is how much a provider's slow_down lengthens the interval,
// as RFC 8628 section 3.5 requires.
const slowDownStep = 5 * time.Second

// defaultInterval is the device polling interval when a driver names none,
// RFC 8628's default.
const defaultInterval = 5 * time.Second

// Completion is what the product's key hook sees when a flow yields a
// credential. Params are the flow's start inputs, such as a connection name.
type Completion struct {
	Caller   core.Caller
	Instance string
	Method   Method
	Params   map[string]string
	Record   tokenstore.Record
}

// Options configure a Service. Store, Credentials, Drivers and CredentialKey
// are required.
type Options struct {
	Store       FlowStore
	Credentials core.CredentialStore
	// Drivers returns the driver that runs method for instance, or an
	// error when the instance does not offer it. It is consulted on every
	// operation, so a product's settings may change between them.
	Drivers func(instance string, method Method) (Driver, error)
	// CredentialKey chooses the key a completed flow's credential is saved
	// under: which credential serves whom is product policy.
	CredentialKey func(ctx context.Context, completion Completion) (string, error)
	// Now is the clock; nil means time.Now. It must match the store's.
	Now func() time.Time
}

// Service runs OAuth flows for a product. The product keeps its HTTP routes
// and maps each onto one Service call; see the package documentation.
type Service struct {
	store       FlowStore
	credentials core.CredentialStore
	drivers     func(string, Method) (Driver, error)
	key         func(context.Context, Completion) (string, error)
	now         func() time.Time
}

// New validates options and returns a Service.
func New(options Options) (*Service, error) {
	if options.Store == nil || options.Credentials == nil || options.Drivers == nil || options.CredentialKey == nil {
		return nil, errors.New("oauthflow: Store, Credentials, Drivers and CredentialKey are required")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &Service{
		store: options.Store, credentials: options.Credentials,
		drivers: options.Drivers, key: options.CredentialKey, now: now,
	}, nil
}

// StartOption adds a product input to Start.
type StartOption func(*StartRequest)

// WithRedirectURI sets the product's callback for a browser flow.
func WithRedirectURI(redirectURI string) StartOption {
	return func(request *StartRequest) { request.RedirectURI = strings.TrimSpace(redirectURI) }
}

// WithParams passes start inputs to the driver and, at completion, to the
// key hook. They stay on the server and may hold secrets.
func WithParams(params map[string]string) StartOption {
	return func(request *StartRequest) { request.Params = maps.Clone(params) }
}
