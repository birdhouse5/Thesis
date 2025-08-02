class HierarchicalPolicy:
    def __init__(self, obs_dim, latent_dim, action_dim):
        pass
    def forward(self, obs, latent):
        # Step 1: What to do with each asset
        decisions = self.decision_head(obs, latent)
        
        # Step 2: How much weight for chosen actions
        long_weights = self.long_head(obs, latent)
        short_weights = self.short_head(obs, latent)
        
        return decisions, long_weights, short_weights