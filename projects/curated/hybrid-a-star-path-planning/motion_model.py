import math
import numpy as np

def find_route(now_point, ind, step, min_r, safe_dis, ob_coo):
    isok = 0
    theta = now_point[2]
    n2 = 20
    
    if ind == 1:
        cenx = now_point[0] - min_r * math.sin(theta)
        ceny = now_point[1] + min_r * math.cos(theta)
        t_vals = theta - math.pi / 2 + np.linspace(0, step / min_r, n2)
        dx = cenx + min_r * np.cos(t_vals)
        dy = ceny + min_r * np.sin(t_vals)
        angle = t_vals + math.pi / 2
        route = np.column_stack([dx, dy, angle])
        x = dx[-1]
        y = dy[-1]
        sita = angle[-1]
        
        for i in range(n2):
            temp = route[i, 0:2]
            for j in range(len(ob_coo)):
                if np.linalg.norm(temp - ob_coo[j]) < safe_dis + math.sqrt(2) / 2:
                    isok = 1
                    return isok, x, y, sita, route
                    
    elif ind == 2:
        dvec = [step * math.cos(theta), step * math.sin(theta)]
        dx = now_point[0] + np.linspace(0, dvec[0], n2)
        dy = now_point[1] + np.linspace(0, dvec[1], n2)
        angle = np.full(n2, theta)
        route = np.column_stack([dx, dy, angle])
        x = dx[-1]
        y = dy[-1]
        sita = angle[-1]
        
        for i in range(n2):
            temp = route[i, 0:2]
            for j in range(len(ob_coo)):
                if np.linalg.norm(temp - ob_coo[j]) < safe_dis + math.sqrt(2) / 2:
                    isok = 1
                    return isok, x, y, sita, route
                    
    elif ind == 3:
        cenx = now_point[0] + min_r * math.sin(theta)
        ceny = now_point[1] - min_r * math.cos(theta)
        t_vals = theta + math.pi / 2 - np.linspace(0, step / min_r, n2)
        dx = cenx + min_r * np.cos(t_vals)
        dy = ceny + min_r * np.sin(t_vals)
        angle = t_vals - math.pi / 2
        route = np.column_stack([dx, dy, angle])
        x = dx[-1]
        y = dy[-1]
        sita = angle[-1]
        
        for i in range(n2):
            temp = route[i, 0:2]
            for j in range(len(ob_coo)):
                if np.linalg.norm(temp - ob_coo[j]) < safe_dis + math.sqrt(2) / 2:
                    isok = 1
                    return isok, x, y, sita, route
                    
    elif ind == 4:
        cenx = now_point[0] - min_r * math.sin(theta)
        ceny = now_point[1] + min_r * math.cos(theta)
        t_vals = theta - math.pi / 2 - np.linspace(0, step / min_r, n2)
        dx = cenx + min_r * np.cos(t_vals)
        dy = ceny + min_r * np.sin(t_vals)
        angle = t_vals + math.pi / 2
        route = np.column_stack([dx, dy, angle])
        x = dx[-1]
        y = dy[-1]
        sita = angle[-1]
        
        for i in range(n2):
            temp = route[i, 0:2]
            for j in range(len(ob_coo)):
                if np.linalg.norm(temp - ob_coo[j]) < safe_dis + math.sqrt(2) / 2:
                    isok = 1
                    return isok, x, y, sita, route
                    
    elif ind == 5:
        dvec = [step * math.cos(theta), step * math.sin(theta)]
        dx = now_point[0] - np.linspace(0, dvec[0], n2)
        dy = now_point[1] - np.linspace(0, dvec[1], n2)
        angle = np.full(n2, theta)
        route = np.column_stack([dx, dy, angle])
        x = dx[-1]
        y = dy[-1]
        sita = angle[-1]
        
        for i in range(n2):
            temp = route[i, 0:2]
            for j in range(len(ob_coo)):
                if np.linalg.norm(temp - ob_coo[j]) < safe_dis + math.sqrt(2) / 2:
                    isok = 1
                    return isok, x, y, sita, route
                    
    elif ind == 6:
        cenx = now_point[0] + min_r * math.sin(theta)
        ceny = now_point[1] - min_r * math.cos(theta)
        t_vals = theta + math.pi / 2 + np.linspace(0, step / min_r, n2)
        dx = cenx + min_r * np.cos(t_vals)
        dy = ceny + min_r * np.sin(t_vals)
        angle = t_vals - math.pi / 2
        route = np.column_stack([dx, dy, angle])
        x = dx[-1]
        y = dy[-1]
        sita = angle[-1]
        
        for i in range(n2):
            temp = route[i, 0:2]
            for j in range(len(ob_coo)):
                if np.linalg.norm(temp - ob_coo[j]) < safe_dis + math.sqrt(2) / 2:
                    isok = 1
                    return isok, x, y, sita, route
    
    return isok, x, y, sita, route
