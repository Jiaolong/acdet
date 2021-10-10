from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE, PillarVFE_Scale
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'PillarVFE_Scale': PillarVFE_Scale,
    'ImageVFE': ImageVFE
}
