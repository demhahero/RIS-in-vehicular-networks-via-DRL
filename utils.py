class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ProtoConvNet:
    def __init__(self, inChannels, numFilters, filterStridePad, useBatchNorm):
        self.inChannels = inChannels
        self.numFilters = numFilters
        self.filterStridePad = filterStridePad
        self.useBatchNorm = useBatchNorm


class ProtoLSTMNet:
    def __init__(self, numCells):
        self.numCells = numCells


class ProtoMLP:
    def __init__(self, layerSizes, activationFunctions, useBias):
        self.layerSizes = layerSizes
        self.activationFunctions = activationFunctions
        self.useBias = useBias
