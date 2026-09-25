package execution_test

import (
	"context"
	"sync"

	core "github.com/xibodev/llmgw-core"
)

// scripted is a provider whose tries fail with outcomes in turn, then
// with always, which nil makes a success. It counts each operation's
// tries.
type scripted struct {
	mu       sync.Mutex
	outcomes []error
	always   error
	invokes  int
	streams  int
	lists    int
	counts   int
	opened   []*stream
}

func (p *scripted) next(tries *int) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	*tries++
	if len(p.outcomes) > 0 {
		err := p.outcomes[0]
		p.outcomes = p.outcomes[1:]
		return err
	}
	return p.always
}

func (p *scripted) tries() (invokes, streams int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.invokes, p.streams
}

func (p *scripted) NativeSurfaces(string) []core.ModelSurface {
	return []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses}
}

func (p *scripted) Invoke(_ context.Context, _ core.Request) (core.Response, error) {
	if err := p.next(&p.invokes); err != nil {
		return core.Response{}, err
	}
	return core.Response{Body: []byte(`{"ok":true}`), ContentType: core.ContentTypeJSON}, nil
}

// Stream hands back a stream even when opening fails, as some providers
// do, so the wrapper must close it.
func (p *scripted) Stream(_ context.Context, _ core.Request) (core.StreamIter, error) {
	err := p.next(&p.streams)
	opened := &stream{frames: []string{done}}
	p.mu.Lock()
	p.opened = append(p.opened, opened)
	p.mu.Unlock()
	return opened, err
}

// ListModels always answers, leaving outcomes to the guarded operations.
func (p *scripted) ListModels(context.Context, *core.Credential) ([]core.ModelInfo, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.lists++
	return []core.ModelInfo{{ID: "model"}}, nil
}

// counter is a scripted provider that also counts tokens natively.
type counter struct{ *scripted }

func (c counter) CountTokens(context.Context, core.TokenCountRequest) (core.TokenCount, error) {
	if err := c.next(&c.counts); err != nil {
		return core.TokenCount{}, err
	}
	return core.TokenCount{InputTokens: 7}, nil
}

// decorator hands every request to the provider it wraps, as
// translation.Adapter does for its native surfaces.
type decorator struct{ core.Provider }

func (d decorator) Unwrap() core.Provider { return d.Provider }

// Requests of each surface the parity tests send.
func request(surface core.ModelSurface, body string) core.Request {
	return core.Request{Surface: surface, Model: "model", Body: []byte(body), ContentType: core.ContentTypeJSON}
}

var (
	chatRequest      = request(core.ModelSurfaceChatCompletions, `{"messages":[]}`)
	responsesRequest = request(core.ModelSurfaceResponses, `{"input":"hello"}`)
	messagesRequest  = request(core.ModelSurfaceMessages, `{"messages":[]}`)
)
