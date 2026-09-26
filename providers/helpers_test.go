package providers

import core "github.com/xibodev/llmgw-core"

func lossPaths(losses []core.Loss) []string {
	paths := make([]string, len(losses))
	for index, loss := range losses {
		paths[index] = string(loss.Severity) + " " + string(loss.Class) + " " + loss.Path
	}
	return paths
}

func modelIDsOf(models []core.ModelInfo) []string {
	ids := make([]string, len(models))
	for i, model := range models {
		ids[i] = model.ID
	}
	return ids
}
