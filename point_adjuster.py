import numpy as np
from config import GRID_PARAMS

class PointAdjuster:
    def __init__(self):
        self.x_spacing = GRID_PARAMS['x_spacing']
        self.z_spacing = GRID_PARAMS['z_spacing']

    def generate_grid_points(self, ref_point):
        origin_x = ref_point[0] - 2 * self.x_spacing
        origin_z = ref_point[1] + self.z_spacing
        grid_points = np.array([
            [origin_x + j * self.x_spacing, origin_z - i * self.z_spacing]
            for i in range(GRID_PARAMS['row_count'])
            for j in range(GRID_PARAMS['col_count'])
        ])
        return grid_points, origin_x, origin_z

    def fit_line(self, x, z):
        if len(x) < 2:
            return 0, z.mean()
        coeffs = np.polyfit(x, z, 1)
        return coeffs[0], coeffs[1]

    def adjust_points(self, points_3d, points_2d, x_labels, z_labels, origin_x, origin_z, weight=0.5):
        adjusted_2d = points_2d.copy()
        row_count = GRID_PARAMS['row_count']
        col_count = GRID_PARAMS['col_count']

        for i in range(row_count):
            mask = z_labels == i
            points_in_row = mask.sum()
            if points_in_row > 0:
                row_x = points_2d[mask, 0]
                row_z = points_2d[mask, 1]
                slope, intercept = self.fit_line(row_x, row_z)
                current_z_mean = row_z.mean()
                target_z = origin_z - i * self.z_spacing
                z_shift = weight * (target_z - current_z_mean)
                adjusted_2d[mask, 1] = row_x * slope + intercept + z_shift

        for j in range(col_count):
            mask = x_labels == j
            points_in_col = mask.sum()
            if points_in_col > 0:
                target_x = origin_x + j * self.x_spacing
                current_x_mean = adjusted_2d[mask, 0].mean()
                x_shift = weight * (target_x - current_x_mean)
                adjusted_2d[mask, 0] += x_shift

        return np.column_stack((adjusted_2d[:, 0], points_3d[:, 1], adjusted_2d[:, 1]))