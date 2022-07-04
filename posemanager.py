class PoseManager():
    def __init__(self, poselist):
        self.poselist = poselist
    
    def __getitem__(self, key):
        return self.poselist[key]
    
    def __setitem__(self, key, value):
        self.poselist[key] = value

    def update(self, seq, T_10, loss):
        self.poselist[seq, 0] = T_10
        self.poselist[seq, 1, 0] = loss