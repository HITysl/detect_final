import numpy as np
from collections import defaultdict
import pyads
from config import GRID_PARAMS

class Box:
    def __init__(self, id, row, col, coords_3d, width_3d, height_3d, side="unknown"):
        self.id = id
        self.row = row
        self.col = col
        self.aGraspPoint_Side = coords_3d
        self.width_3d = width_3d
        self.height_3d = height_3d
        self.side = side
        self.aGraspPoint_Top = np.array([
            self.aGraspPoint_Side[0],
            self.aGraspPoint_Side[1]+int(GRID_PARAMS['y_spacing']*500),
            self.aGraspPoint_Side[2]+int(GRID_PARAMS['z_spacing']*500)
        ])

    def __str__(self):
        return (f"Box(id={self.id}, row={self.row}, col={self.col}, side={self.side}, "
                f"aGraspPoint_Side=[{self.aGraspPoint_Side[0]:.3f}, {self.aGraspPoint_Side[1]:.3f}, {self.aGraspPoint_Side[2]:.3f}], "
                f"aGraspPoint_Top=[{self.aGraspPoint_Top[0]:.3f}, {self.aGraspPoint_Top[1]:.3f}, {self.aGraspPoint_Top[2]:.3f}], "
                f"width_3d={self.width_3d:.3f}, height_3d={self.height_3d:.3f})")

class Tasks:
    def __init__(self, boxes, total_rows, total_cols, box_account):
        self.nTotalRow = total_rows
        self.nTotalCol = total_cols
        self.flag = True
        self.all_box_origin=boxes
        self.box_account = box_account
        # 检查每行箱子数量是否为6
        # row_counts = defaultdict(int)
        # for box in boxes:
        #     row_counts[box.row] += 1
        # for row in range(1, total_rows + 1):
        #     if row_counts[row] != 6:
        #         raise ValueError(f"Row {row} has {row_counts[row]} boxes, expected 6")

        self.aLeftBoxArray = self._reorder_boxes(
                [box for box in boxes if box.side == 'left']
            )
        self.aRightBoxArray = self._reorder_boxes(
                [box for box in boxes if box.side == 'right']
            )

        self.nLeftBoxCount = len(self.aLeftBoxArray)
        self.nRightBoxCount = len(self.aRightBoxArray)
        self.aHeightEachRow = self._compute_row_avg_heights(boxes)

    def _compute_row_avg_heights(self, boxes):
        row_heights = [[] for _ in range(self.nTotalRow)]
        for box in boxes:
            row_idx = box.row - 1
            if 0 <= row_idx < self.nTotalRow:
                row_heights[row_idx].append(box.aGraspPoint_Side[2])
        return [np.mean(heights) if heights else None for heights in row_heights]

    def _reorder_boxes(self, boxes):
        """
        按行分组，左箱子按col顺序[2,3,1]（id按2,3,1），右箱子按col顺序[5,6,4]（id按5,6,4）。
        """
        if not boxes:
            return []

        # 按行分组
        row_groups = defaultdict(list)
        for box in boxes:
            row_groups[box.row].append(box)

        # 定义排序顺序
        left_col_order = [2, 3, 1]  # 左箱子：中间col=2,右col=3,左col=1
        right_col_order = [5, 6, 4]  # 右箱子：中间col=5,右col=6,左col=4

        sorted_boxes = []
        for row in sorted(row_groups.keys()):
            row_boxes = row_groups[row]
            # 根据side选择排序顺序
            col_order = left_col_order if row_boxes[0].side == 'left' else right_col_order
            # 按col_order排序
            sorted_row = sorted(
                row_boxes,
                key=lambda b: col_order.index(b.col) if b.col in col_order else len(col_order)
            )
            sorted_boxes.extend(sorted_row)
        return sorted_boxes

    def __str__(self):
        row_heights_str = ", ".join([f"Row {i + 1}: {h:.3f}" if h is not None else f"Row {i + 1}: None"
                                     for i, h in enumerate(self.aHeightEachRow)])
        return (f"Tasks:\n"
                f"  nLeftBoxCount: {self.nLeftBoxCount}\n"
                f"  nRightBoxCount: {self.nRightBoxCount}\n"
                f"  nTotalRow: {self.nTotalRow}\n"
                f"  nTotalCol: {self.nTotalCol}\n"
                f"  aLeftBoxArray: {self.aLeftBoxArray}\n"
                f"  aRightBoxArray: {self.aRightBoxArray}\n"
                f"  aHeightEachRow: [{row_heights_str}]")

# utils.py 部分
def transmit_to_plc(tasks):
    plc = pyads.Connection("192.168.1.20.1.1", 851)
    plc.open()

    nbox_l = int(GRID_PARAMS['x_spacing'])*1000
    nbox_w = int(GRID_PARAMS['y_spacing'])*1000
    nbox_h = int(GRID_PARAMS['z_spacing'])*1000

    nLeftBoxCount = int(tasks.nLeftBoxCount)
    nRightBoxCount = int(tasks.nRightBoxCount)
    nTotalRow = int(tasks.nTotalRow)
    nTotalCol = int(tasks.nTotalCol)

    leftArm_Data = []
    rightArm_Data = []

    for box in tasks.aLeftBoxArray:
        flat_data = [
            int(box.id),
            int(box.row),
            int(box.col),
            int(box.aGraspPoint_Top[0]),
            int(box.aGraspPoint_Top[1]),
            int(box.aGraspPoint_Top[2]),
            int(box.aGraspPoint_Side[0]),
            int(box.aGraspPoint_Side[1]),
            int(box.aGraspPoint_Side[2])
        ]
        leftArm_Data.extend(flat_data)

    # 打印左箱子ID顺序
    print("Left Box IDs:", [box.id for box in tasks.aLeftBoxArray])

    for box in tasks.aRightBoxArray:
        flat_data = [
            int(box.id),
            int(box.row),
            int(box.col),
            int(box.aGraspPoint_Top[0]),
            int(box.aGraspPoint_Top[1]),
            int(box.aGraspPoint_Top[2]),
            int(box.aGraspPoint_Side[0]),
            int(box.aGraspPoint_Side[1]),
            int(box.aGraspPoint_Side[2])
        ]
        rightArm_Data.extend(flat_data)

    # 打印右箱子ID顺序
    print("Right Box IDs:", [box.id for box in tasks.aRightBoxArray])

    aHeightEachRow = [int(h) if h is not None else 0 for h in tasks.aHeightEachRow]
    try:
        plc.write_by_name('Camera.nbox_l', nbox_l, pyads.PLCTYPE_UINT)
        plc.write_by_name('Camera.nbox_w', nbox_w, pyads.PLCTYPE_UINT)
        plc.write_by_name('Camera.nbox_h', nbox_h, pyads.PLCTYPE_UINT)
        plc.write_by_name('Camera.nLeftBoxCount', nLeftBoxCount, pyads.PLCTYPE_UINT)
        plc.write_by_name('Camera.nRightBoxCount', nRightBoxCount, pyads.PLCTYPE_UINT)
        plc.write_by_name('Camera.nTotalRow', nTotalRow, pyads.PLCTYPE_UINT)
        plc.write_by_name('Camera.nTotalCol', nTotalCol, pyads.PLCTYPE_UINT)
        plc.write_by_name('Camera.aHeightEachRow', aHeightEachRow, pyads.PLCTYPE_ARR_INT(nTotalRow))
        plc.write_by_name('Camera.aLeftBoxArrayFlat', leftArm_Data,
                          pyads.PLCTYPE_ARR_INT(nLeftBoxCount * 9))
        plc.write_by_name('Camera.aRightBoxArrayFlat', rightArm_Data,
                          pyads.PLCTYPE_ARR_INT(nRightBoxCount * 9))
        print("Data successfully written to PLC")
    except pyads.ADSError as e:
        print(f"PLC write error: {e}")
    finally:
        plc.close()
        print("PLC connection closed")
