from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .yolox_head_single import YOLOXHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_point_head import CenterHead
from .yolox_head_single_split import YOLOXHeadSingleSplit
from .yolox_head_single_split_v2 import YOLOXHeadSingleSplitV2

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'YOLOXHeadSingle': YOLOXHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'YOLOXHeadSingleSplit':YOLOXHeadSingleSplit,
    'YOLOXHeadSingleSplitV2':YOLOXHeadSingleSplitV2
}
