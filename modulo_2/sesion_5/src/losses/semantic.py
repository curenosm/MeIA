import torch
import torch.nn.functional as F


def Semantic(Xs, Xt, Ys, Yt, Cs_memory, Ct_memory, decay=0.3):
    # Clone memory
    Cs = Cs_memory.clone()
    Ct = Ct_memory.clone()

    # spherical normalization
    r = torch.norm(Xs, dim=1)[0]
    Ct = r * Ct / (torch.norm(Ct, dim=1, keepdim=True)+1e-10)
    Cs = r * Cs / (torch.norm(Cs, dim=1, keepdim=True)+1e-10)

    K = Cs.size(0)
    # for each class
    for k in range(K):

        # Get samples from class 'k'
        Xs_k = Xs[Ys==k]    # source domain
        Xt_k = Xt[Yt==k]    # target domain

        # validate if there is zero elements in Source domain
        if len(Xs_k)==0:
            Cs_k = 0.0
        else:
            # get mean from elements of the source domain
            Cs_k = torch.mean(Xs_k,dim=0)

        # validate if there is zero elements in Target domain
        if len(Xt_k) == 0:
            Ct_k = 0.0
        else:
            # get mean from elements of the target domain
            Ct_k = torch.mean(Xt_k,dim=0)

        # Moving average (MA)
        Cs[k, :] = (1 - decay) * Cs_memory[k, :] + decay * Cs_k
        Ct[k, :] = (1 - decay) * Ct_memory[k, :] + decay * Ct_k

    Dist = cosine_matrix(Cs, Ct)

    return torch.sum(torch.diag(Dist)), Cs, Ct

def cosine_matrix(x,y):
    x=F.normalize(x,dim=1)
    y=F.normalize(y,dim=1)
    xty=torch.sum(x.unsqueeze(1)*y.unsqueeze(0),2)
    return 1-xty
