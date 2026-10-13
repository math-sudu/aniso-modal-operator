import math

def arc_point(segment, theta):
    cx, cy = segment['center']
    radius = segment['radius']
    return [cx + radius * math.cos(theta), cy + radius * math.sin(theta)]

def arc_angles(segment, levels=()):
    start = segment['start_rad']
    end = start + segment['sweep_rad']
    angles = [start, end]
    angles += [k * math.pi / 2 for k in range(5) if start < k * math.pi / 2 < end]
    for y in levels:
        sine = (y - segment['center'][1]) / segment['radius']
        if abs(sine) <= 1:
            base = math.asin(sine)
            angles += [a for a in (base, math.pi - base, base + 2 * math.pi) if start < a < end]
    unique = []
    for angle in sorted(angles):
        if not unique or angle - unique[-1] > 1e-12:
            unique.append(angle)
    return unique

def sample(segments, step, levels=()):
    points, tags = ([], [])
    for segment in segments:
        cuts = arc_angles(segment, levels)
        piece = []
        for start, end in zip(cuts, cuts[1:]):
            count = max(1, math.ceil((end - start) / step))
            part = [arc_point(segment, start + (end - start) * j / count) for j in range(count + 1)]
            piece.extend(part if not piece else part[1:])
        if points:
            if math.dist(points[-1], piece[0]) > 1e-10:
                raise ValueError('Disconnected tangent arcs')
            piece = piece[1:]
        points.extend(piece)
        tags.extend([segment['arc_id']] * len(piece))
    return (points, tags)
