"""Candidate features computed on the model device.

Grid geometry combines float32 additions with float64 scalar arithmetic.
Device transcendental functions and reductions can differ from NumPy by
floating-point rounding."""
import math
import torch
from torch.nn import functional as F


def mode_top5(p):
    modes = (p >= F.pad(p[:, :-1], (1, 0), value=-1.)) & (p >= F.pad(p[:, 1:], (0, 1), value=-1.))
    top = torch.zeros_like(modes).scatter_(1, p.argsort(dim=-1)[:, -5:], True)
    return modes | top


@torch.no_grad()
def build_gpu_candidates(grid, ge, supports, rr, *, stable=True, prepared=None):
    from .heads import FINE_OFFSETS, WIDE_OFFSETS
    from .networks import flatten_supports
    device = grid.device
    if prepared is None:
        sb, si = flatten_supports(supports, device)
        counts = torch.tensor([len(s) for s in supports], device=device)
        starts = counts.cumsum(0)-counts
        ranks = torch.arange(len(sb), device=device)-starts[sb]
        fine_offsets = torch.as_tensor(FINE_OFFSETS, device=device)
        wide_offsets = torch.as_tensor(WIDE_OFFSETS, device=device)
    else:
        sb, si, counts, ranks, fine_offsets, wide_offsets = (
            prepared[name] for name in ("sb", "si", "counts", "ranks", "fine_offsets", "wide_offsets"))
    base, centre, cond = rr['base_angle'], rr['route_base_angle'], rr['conditional_angle']
    angles = torch.cat((cond[:,None], base[:,None]+fine_offsets, centre[:,None]+wide_offsets), 1)
    a = angles.double()
    gi = torch.round((a-grid[0].double())/(grid[1]-grid[0]).double()).long().clamp(0,len(grid)-1)
    if len(grid) == 281:
        # Grid offsets use the 0.5-degree coordinate convention.
        ids = (gi[:,:,None]+torch.arange(-3,4,device=device)).clamp(0,len(grid)-1)
        gw = ge['location_logits'][sb[:,None,None],ids]
    else:
        # Seven samples still describe +/-1.5 degrees at half-degree spacing.
        # Interpolation does not create extra local peaks or extra routes.
        physical_offsets = torch.arange(-3,4,device=device,dtype=torch.float64) * .5
        coordinate = (gi[:,:,None].double() + physical_offsets / (grid[1]-grid[0]).double()).clamp(0,len(grid)-1)
        lo = coordinate.floor().long()
        hi = (lo+1).clamp_max(len(grid)-1)
        blend = (coordinate-lo).to(ge['location_logits'].dtype)
        low = ge['location_logits'][sb[:,None,None],lo]
        high = ge['location_logits'][sb[:,None,None],hi]
        gw = low + blend*(high-low)
    gp = 1./(1.+(-gw.clamp(-20.,20.)).exp())
    p = ge['location_logits'][sb,si].sigmoid().double()
    fp = rr['fine_probability']
    wz = rr['wide_logits']
    we = (wz-wz.amax(-1,keepdim=True)).exp()
    wp = we/we.sum(-1,keepdim=True).clamp_min(1.e-8)
    fm, wm = mode_top5(fp), mode_top5(wp)
    fj = (fine_offsets[None,None]-(a-base.double()[:,None]).float()[:,:,None]).abs().argmin(-1)
    wj = (wide_offsets[None,None]-(a-centre.double()[:,None]).float()[:,:,None]).abs().argmin(-1)
    fpad, wpad = F.pad(fp,(1,1)), F.pad(wp,(1,1))
    fa, wa = fp.gather(1,fj), wp.gather(1,wj)
    fprom = fa.double()-torch.maximum(fpad.gather(1,fj),fpad.gather(1,fj+2)).double()
    wprom = wa.double()-torch.maximum(wpad.gather(1,wj),wpad.gather(1,wj+2)).double()
    fh = -(fp*fp.clamp_min(1.e-8).log()).sum(-1).double()/math.log(15.)
    wh = -(wp*wp.clamp_min(1.e-8).log()).sum(-1).double()/math.log(15.)
    support_angles = grid[si]
    left = torch.where(ranks>0,support_angles-support_angles.roll(1),10.).clamp_max(10.).double()/10.
    right = torch.where(ranks<counts[sb]-1,support_angles.roll(-1)-support_angles,10.).clamp_max(10.).double()/10.
    def expand(v):
        return v.double()[:,None].expand(-1,31)
    slot = torch.arange(31,device=device)
    scalars = torch.stack((a/70.,expand(si.double()/float(len(grid)-1)),expand(counts[sb].double()/20.),expand(p),
        expand((p.clamp_min(1.e-6)/(1.-p).clamp_min(1.e-6)).log()),expand(cond.double()/70.),
        expand(base.double()/70.),expand(centre.double()/70.),(a-cond.double()[:,None])/1.5,
        (a-cond.double()[:,None]).abs()/1.5,a-base.double()[:,None],fa.double(),fprom,
        expand(fp.amax(-1)),expand(fh),wa.double(),wprom,expand(wp.amax(-1)),expand(wh),
        expand(fm.sum(-1).double()/5.),expand(wm.sum(-1).double()/5.),expand(left),expand(right),
        gw[:,:,3].double(),gp[:,:,3].double(),2.*gw[:,:,3].double()-gw[:,:,2].double()-gw[:,:,4].double(),
        expand(ranks.double()/20.),(slot==0).double()[None].expand(len(sb),-1),
        ((slot>=1)&(slot<=15)).double()[None].expand(len(sb),-1),
        (slot>=16).double()[None].expand(len(sb),-1)),dim=-1).float()
    features = torch.cat((gw,gp,scalars),dim=-1)
    keep = torch.ones_like(angles,dtype=torch.bool) if stable else torch.cat((torch.ones((len(sb),1),device=device,dtype=torch.bool),fm,wm),dim=-1)
    return angles, features, keep, sb, si
