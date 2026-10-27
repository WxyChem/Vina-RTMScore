"""Pinned official RTMScore model1 architecture and checkpoint loading."""
def load_model(root,device):
    import torch
    from RTMScore.model.model2 import RTMScore,DGLGraphTransformer
    def graph(edge):
        return DGLGraphTransformer(in_channels=41,edge_features=edge,num_hidden_channels=128,
            activ_fn=torch.nn.SiLU(),transformer_residual=True,num_attention_heads=4,
            norm_to_apply='batch',dropout_rate=.15,num_layers=6)
    model=RTMScore(graph(10),graph(5),in_channels=128,hidden_dim=128,n_gaussians=10,
                   dropout_rate=.1,dist_threhold=5).to(device)
    checkpoint=root/'trained_models/rtmscore_model1.pth'
    model.load_state_dict(torch.load(str(checkpoint),map_location=torch.device(device))['model_state_dict'])
    model.eval()
    return model,checkpoint
