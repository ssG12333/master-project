import math

class RSPath:
    TYPES = [
        ['L', 'R', 'L', 'N', 'N'],
        ['R', 'L', 'R', 'N', 'N'],
        ['L', 'R', 'L', 'R', 'N'],
        ['R', 'L', 'R', 'L', 'N'],
        ['L', 'R', 'S', 'L', 'N'],
        ['R', 'L', 'S', 'R', 'N'],
        ['L', 'S', 'R', 'L', 'N'],
        ['R', 'S', 'L', 'R', 'N'],
        ['L', 'R', 'S', 'R', 'N'],
        ['R', 'L', 'S', 'L', 'N'],
        ['R', 'S', 'R', 'L', 'N'],
        ['L', 'S', 'L', 'R', 'N'],
        ['L', 'S', 'R', 'N', 'N'],
        ['R', 'S', 'L', 'N', 'N'],
        ['L', 'S', 'L', 'N', 'N'],
        ['R', 'S', 'R', 'N', 'N'],
        ['L', 'R', 'S', 'L', 'R'],
        ['R', 'L', 'S', 'R', 'L']
    ]
    
    def __init__(self, path_type=None, t=0, u=0, v=0, w=0, x=0):
        if path_type is None:
            self.type = ['N', 'N', 'N', 'N', 'N']
        else:
            self.type = path_type
        self.t = t
        self.u = u
        self.v = v
        self.w = w
        self.x = x
        self.total_length = sum(abs(v) for v in [t, u, v, w, x])

def mod2pi(x):
    v = x % (2 * math.pi)
    if v < -math.pi:
        v += 2 * math.pi
    elif v > math.pi:
        v -= 2 * math.pi
    return v

def tauOmega(u, v, xi, eta, phi):
    delta = mod2pi(u - v)
    A = math.sin(u) - math.sin(delta)
    B = math.cos(u) - math.cos(delta) - 1
    t1 = math.atan2(eta * A - xi * B, xi * A + eta * B)
    t2 = 2 * (math.cos(delta) - 2 * math.cos(v) - 2 * math.cos(u)) + 3
    if t2 < 0:
        tau = mod2pi(t1 + math.pi)
    else:
        tau = mod2pi(t1)
    omega = mod2pi(tau - u + v - phi)
    return tau, omega

def LpSpLp(x, y, phi):
    t, u = math.atan2(y - 1 + math.cos(phi), x - math.sin(phi)), \
           math.sqrt((x - math.sin(phi))**2 + (y - 1 + math.cos(phi))**2)
    if t >= 0:
        v = mod2pi(phi - t)
        if v >= 0:
            return True, t, u, v
    return False, 0, 0, 0

def LpSpRp(x, y, phi):
    t1, u1 = math.atan2(y - 1 - math.cos(phi), x + math.sin(phi)), \
             math.sqrt((x + math.sin(phi))**2 + (y - 1 - math.cos(phi))**2)
    if u1**2 >= 4:
        u = math.sqrt(u1**2 - 4)
        theta = math.atan2(2, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)
        if t >= 0 and v >= 0:
            return True, t, u, v
    return False, 0, 0, 0

def LpRmL(x, y, phi):
    xi = x - math.sin(phi)
    eta = y - 1 + math.cos(phi)
    theta, u1 = math.atan2(eta, xi), math.sqrt(xi**2 + eta**2)
    if u1 <= 4:
        u = -2 * math.asin(u1 / 4)
        t = mod2pi(theta + u / 2 + math.pi)
        v = mod2pi(phi - t + u)
        if t >= 0 and u <= 0:
            return True, t, u, v
    return False, 0, 0, 0

def LpRmSmLm(x, y, phi):
    xi = x - math.sin(phi)
    eta = y - 1 + math.cos(phi)
    theta, rho = math.atan2(eta, xi), math.sqrt(xi**2 + eta**2)
    if rho >= 2:
        r = math.sqrt(rho**2 - 4)
        u = 2 - rho
        t = mod2pi(theta + math.atan2(r, -2))
        v = mod2pi(phi - math.pi / 2 - t)
        if t >= 0 and u <= 0 and v <= 0:
            return True, t, u, v
    return False, 0, 0, 0

def LpRmSmRm(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1 - math.cos(phi)
    theta, rho = math.atan2(xi, -eta), math.sqrt((-eta)**2 + xi**2)
    if rho >= 2:
        t = theta
        u = 2 - rho
        v = mod2pi(t + math.pi / 2 - phi)
        if t >= 0 and u <= 0 and v <= 0:
            return True, t, u, v
    return False, 0, 0, 0

def LpRupLumRm(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1 - math.cos(phi)
    rho = (2 + math.sqrt(xi**2 + eta**2)) / 4
    if rho <= 1:
        u = math.acos(rho)
        t, v = tauOmega(u, -u, xi, eta, phi)
        if t >= 0 and v <= 0:
            return True, t, u, v
    return False, 0, 0, 0

def LpRumLumRp(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1 - math.cos(phi)
    rho = (20 - xi**2 - eta**2) / 16
    if 0 <= rho <= 1:
        u = -math.acos(rho)
        if u >= -math.pi / 2:
            t, v = tauOmega(u, u, xi, eta, phi)
            if t >= 0 and v >= 0:
                return True, t, u, v
    return False, 0, 0, 0

def LpRmSLmRp(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1 - math.cos(phi)
    rho = math.sqrt(xi**2 + eta**2)
    if rho >= 2:
        u = 4 - math.sqrt(rho**2 - 4)
        if u <= 0:
            t = mod2pi(math.atan2((4 - u) * xi - 2 * eta, -2 * xi + (u - 4) * eta))
            v = mod2pi(t - phi)
            if t >= 0 and v >= 0:
                return True, t, u, v
    return False, 0, 0, 0

def CSC(x, y, phi):
    Lmin = float('inf')
    path = RSPath()
    isok = False
    
    for sx, sy, sphi, t1, t2, r1, r2 in [
        (x, y, phi, 15, 16, 13, 14),
        (-x, y, -phi, 15, 16, 13, 14),
        (x, -y, -phi, 16, 16, 14, 14),
        (-x, -y, phi, 16, 16, 14, 14)
    ]:
        ok, t, u, v = LpSpLp(sx, sy, sphi)
        if ok:
            L = abs(t) + abs(u) + abs(v)
            if L < Lmin:
                Lmin = L
                path = RSPath(RSPath.TYPES[t1-1], t, u, v, 0, 0)
                isok = True
        
        ok, t, u, v = LpSpRp(sx, sy, sphi)
        if ok:
            L = abs(t) + abs(u) + abs(v)
            if L < Lmin:
                Lmin = L
                path = RSPath(RSPath.TYPES[r1-1], t, u, v, 0, 0)
                isok = True
    
    return isok, path

def CCC(x, y, phi):
    Lmin = float('inf')
    path = RSPath()
    isok = False
    
    for sx, sy, sphi, t1, t2 in [
        (x, y, phi, 1, 2),
        (-x, y, -phi, 1, 2),
        (x, -y, -phi, 2, 2),
        (-x, -y, phi, 2, 2)
    ]:
        ok, t, u, v = LpRmL(sx, sy, sphi)
        if ok:
            L = abs(t) + abs(u) + abs(v)
            if L < Lmin:
                Lmin = L
                path = RSPath(RSPath.TYPES[t1-1], t, u, v, 0, 0)
                isok = True
    
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
    
    for sx, sy, sphi, t1, t2 in [
        (xb, yb, phi, 1, 2),
        (-xb, yb, -phi, 1, 2),
        (xb, -yb, -phi, 2, 2),
        (-xb, -yb, phi, 2, 2)
    ]:
        ok, t, u, v = LpRmL(sx, sy, sphi)
        if ok:
            L = abs(t) + abs(u) + abs(v)
            if L < Lmin:
                Lmin = L
                path = RSPath(RSPath.TYPES[t1-1], v, u, t, 0, 0)
                isok = True
    
    return isok, path

def CCCC(x, y, phi):
    Lmin = float('inf')
    path = RSPath()
    isok = False
    
    for sx, sy, sphi, t1, t2, t3, t4 in [
        (x, y, phi, 3, 4, 3, 4),
        (-x, y, -phi, 3, 4, 3, 4),
        (x, -y, -phi, 4, 4, 4, 4),
        (-x, -y, phi, 4, 4, 4, 4)
    ]:
        ok, t, u, v = LpRupLumRm(sx, sy, sphi)
        if ok:
            L = abs(t) + 2 * abs(u) + abs(v)
            if L < Lmin:
                Lmin = L
                path = RSPath(RSPath.TYPES[t1-1], t, u, -u, v, 0)
                isok = True
        
        ok, t, u, v = LpRumLumRp(sx, sy, sphi)
        if ok:
            L = abs(t) + 2 * abs(u) + abs(v)
            if L < Lmin:
                Lmin = L
                path = RSPath(RSPath.TYPES[t1-1], t, u, u, v, 0)
                isok = True
    
    return isok, path

def CCSC(x, y, phi):
    Lmin = float('inf')
    path = RSPath()
    isok = False
    
    for sx, sy, sphi, t1, t2 in [
        (x, y, phi, 5, 6),
        (-x, y, -phi, 5, 6),
        (x, -y, -phi, 6, 6),
        (-x, -y, phi, 6, 6)
    ]:
        ok, t, u, v = LpRmSmLm(sx, sy, sphi)
        if ok:
            L = abs(t) + abs(u) + abs(v)
            if L < Lmin:
                Lmin = L
                path = RSPath(RSPath.TYPES[t1-1], t, -math.pi/2, u, v, 0)
                isok = True
        
        ok, t, u, v = LpRmSmRm(sx, sy, sphi)
        if ok:
            L = abs(t) + abs(u) + abs(v)
            if L < Lmin:
                Lmin = L
                path = RSPath(RSPath.TYPES[t1-1], t, -math.pi/2, u, v, 0)
                isok = True
    
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
    
    for sx, sy, sphi, t1, t2 in [
        (xb, yb, phi, 7, 8),
        (-xb, yb, -phi, 7, 8),
        (xb, -yb, -phi, 8, 8),
        (-xb, -yb, phi, 8, 8)
    ]:
        ok, t, u, v = LpRmSmLm(sx, sy, sphi)
        if ok:
            L = abs(t) + abs(u) + abs(v)
            if L < Lmin:
                Lmin = L
                path = RSPath(RSPath.TYPES[t1-1], v, u, -math.pi/2, t, 0)
                isok = True
        
        ok, t, u, v = LpRmSmRm(sx, sy, sphi)
        if ok:
            L = abs(t) + abs(u) + abs(v)
            if L < Lmin:
                Lmin = L
                path = RSPath(RSPath.TYPES[t1-1], v, u, -math.pi/2, t, 0)
                isok = True
    
    return isok, path

def CCSCC(x, y, phi):
    Lmin = float('inf')
    path = RSPath()
    isok = False
    
    for sx, sy, sphi, t1, t2 in [
        (x, y, phi, 17, 18),
    ]:
        ok, t, u, v = LpRmSLmRp(sx, sy, sphi)
        if ok:
            L = abs(t) + abs(u) + abs(v)
            if L < Lmin:
                Lmin = L
                path = RSPath(RSPath.TYPES[t1-1], t, -math.pi/2, u, -math.pi/2, v)
                isok = True
    
    return isok, path

def find_rs_path(goalPose, rmin):
    x = goalPose[0] / rmin
    y = goalPose[1] / rmin
    phi = goalPose[2]
    
    results = []
    isok1, path1 = CSC(x, y, phi)
    isok2, path2 = CCC(x, y, phi)
    isok3, path3 = CCCC(x, y, phi)
    isok4, path4 = CCSC(x, y, phi)
    isok5, path5 = CCSCC(x, y, phi)
    
    isoks = [isok1, isok2, isok3, isok4, isok5]
    paths = [path1, path2, path3, path4, path5]
    
    Lmin = float('inf')
    path = None
    
    for i in range(5):
        if isoks[i]:
            if Lmin > paths[i].total_length:
                Lmin = paths[i].total_length
                path = paths[i]
    
    return path

def reeds_shepp(startPose, goalPose, min_r):
    x = goalPose[0] - startPose[0]
    y = goalPose[1] - startPose[1]
    
    goal_local = [0, 0, 0]
    goal_local[0] = x * math.cos(startPose[2]) + y * math.sin(startPose[2])
    goal_local[1] = -x * math.sin(startPose[2]) + y * math.cos(startPose[2])
    goal_local[2] = goalPose[2] - startPose[2]
    
    if goal_local[2] > math.pi:
        goal_local[2] -= 2 * math.pi
    elif goal_local[2] <= -math.pi:
        goal_local[2] += 2 * math.pi
    
    path = find_rs_path(goal_local, min_r)
    
    if path is None:
        return float('inf'), []
    
    distance = min_r * path.total_length
    route = get_route(startPose, path, min_r)
    
    return distance, route

def get_route(startPose, path, rmin):
    path_type = path.type
    x = []
    y = []
    angle = []
    n = 50
    
    seg = [path.t, path.u, path.v, path.w, path.x]
    pvec = list(startPose)
    
    for i in range(5):
        if path_type[i] == 'S':
            theta = pvec[2]
            if seg[i] == 0:
                continue
            dl = rmin * seg[i]
            dvec = [dl * math.cos(theta), dl * math.sin(theta), 0]
            num_points = int(abs(seg[i]) * n) + 2
            x.extend([pvec[0] + i * dvec[0] / (num_points - 1) for i in range(num_points)])
            y.extend([pvec[1] + i * dvec[1] / (num_points - 1) for i in range(num_points)])
            angle.extend([theta] * num_points)
            pvec = [pvec[0] + dvec[0], pvec[1] + dvec[1], theta]
        elif path_type[i] == 'L':
            theta = pvec[2]
            dtheta = seg[i]
            if dtheta == 0:
                continue
            cenx = pvec[0] - rmin * math.sin(theta)
            ceny = pvec[1] + rmin * math.cos(theta)
            num_points = int(abs(dtheta) * n) + 2
            t_vals = [theta - math.pi/2 + i * dtheta / (num_points - 1) for i in range(num_points)]
            dx = [cenx + rmin * math.cos(t) for t in t_vals]
            dy = [ceny + rmin * math.sin(t) for t in t_vals]
            x.extend(dx)
            y.extend(dy)
            angle.extend([t + math.pi/2 for t in t_vals])
            theta = theta + dtheta
            pvec = [dx[-1], dy[-1], theta]
        elif path_type[i] == 'R':
            theta = pvec[2]
            dtheta = -seg[i]
            if seg[i] == 0:
                continue
            cenx = pvec[0] + rmin * math.sin(theta)
            ceny = pvec[1] - rmin * math.cos(theta)
            num_points = int(abs(seg[i]) * n) + 2
            t_vals = [theta + math.pi/2 + i * dtheta / (num_points - 1) for i in range(num_points)]
            dx = [cenx + rmin * math.cos(t) for t in t_vals]
            dy = [ceny + rmin * math.sin(t) for t in t_vals]
            x.extend(dx)
            y.extend(dy)
            angle.extend([t - math.pi/2 for t in t_vals])
            theta = theta + dtheta
            pvec = [dx[-1], dy[-1], theta]
    
    route = [[x[i], y[i], angle[i]] for i in range(len(x))]
    return route
