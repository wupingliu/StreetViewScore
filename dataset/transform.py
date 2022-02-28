import math as Math
# 坐标转换用函数
pi = 3.14159265358979324
a = 6378245.0
ee = 0.00669342162296594323


# WGS84转火星坐标系
def outOfChina(lon, lat):
    if (lon<72.004 or lon>137.8347) and (lat<0.8293 or lat>55.8271):
        return True
    else:
        return False


def transformLat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * Math.sqrt(abs(x))
    ret += (20.0 * Math.sin(6.0 * x * pi) + 20.0 * Math.sin(2.0 * x * pi)) * 2.0 / 3.0
    ret += (20.0 * Math.sin(y * pi) + 40.0 * Math.sin(y / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * Math.sin(y / 12.0 * pi) + 320 * Math.sin(y * pi / 30.0)) * 2.0 / 3.0
    return ret


def transformLon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * Math.sqrt(abs(x))
    ret += (20.0 * Math.sin(6.0 * x * pi) + 20.0 * Math.sin(2.0 * x * pi)) * 2.0 / 3.0
    ret += (20.0 * Math.sin(x * pi) + 40.0 * Math.sin(x / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * Math.sin(x / 12.0 * pi) + 300.0 * Math.sin(x / 30.0 * pi)) * 2.0 / 3.0
    return ret


def transform(wgLon, wgLat):
    point = [0, 0]
    if outOfChina(wgLon, wgLat):
        point[0] = wgLon
        point[1] = wgLat
        return
    dLat = transformLat(wgLon - 105.0, wgLat - 35.0)
    dLon = transformLon(wgLon - 105.0, wgLat - 35.0)
    radLat = wgLat / 180.0 * pi
    magic = Math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = Math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * Math.cos(radLat) * pi)
    point[0] = wgLon + dLon
    point[1] = wgLat + dLat
    return point


# 火星坐标系转百度坐标系
x_pi = 3.14159265358979324 * 3000.0 / 180.0


def marsTobaidu(mars_point):
    baidu_point = [0, 0]
    x = float(mars_point[0])
    y = float(mars_point[1])
    z = Math.sqrt(x * x + y * y) + 0.00002 * Math.sin(y * x_pi)
    theta = Math.atan2(y, x) + 0.000003 * Math.cos(x * x_pi)
    baidu_point[0] = z * Math.cos(theta) + 0.0065
    baidu_point[1] = z * Math.sin(theta) + 0.006
    return baidu_point


def baiduTomars(baidu_point):
    mars_point = [0, 0]
    x = baidu_point[0]-0.0065
    y = baidu_point[1]-0.006
    z = Math.sqrt(x*x+y*y)- 0.00002 * Math.sin(y * x_pi)
    theta = Math.atan2(y, x) - 0.000003 * Math.cos(x * x_pi)
    mars_point[0] = z * Math.cos(theta)
    mars_point[1] = z * Math.sin(theta)
    return mars_point


# 经纬度坐标转墨卡托投影坐标
def lonlatTomercator(lonlat):
    mercator = [0, 0]
    x = lonlat[0] * 20037508.34 / 180
    y = Math.log(Math.tan((90+lonlat[1])*pi/360))/(pi/180)
    y = y * 20037508.34 / 180
    mercator[0] = x
    mercator[1] = y
    return mercator


# 墨卡托投影坐标转经纬度坐标
def mercatorTolonlat(mercator):
    lonlat = [0, 0]
    x = mercator[0]/20037508.34*180
    y = mercator[1]/20037508.34*180
    y= 180/pi*(2*Math.atan(Math.exp(y*pi/180))-pi/2)
    lonlat[0] = x
    lonlat[1] = y
    return lonlat


def latLng2WebMercator(point):
    earthRad = 6378137.0
    x = point[0] * pi / 180 * earthRad
    a = point[1] * pi / 180
    y = earthRad / 2 * Math.log((1.0 + Math.sin(a)) / (1.0 - Math.sin(a)))
    return [x,y]


'''
 百度墨卡托与百度经纬度转换
'''

MCBAND = [12890594.86, 8362377.87, 5591021, 3481989.83, 1678043.12, 0]
LLBAND = [75.0, 60, 45, 30.0, 15.0, 0.0]
MC2LL = [[1.410526172116255e-8, 0.00000898305509648872, -1.9939833816331, 200.9824383106796, -187.2403703815547, 91.6087516669843, -23.38765649603339, 2.57121317296198, -0.03801003308653, 17337981.2]
    , [-7.435856389565537e-9, 0.000008983055097726239, -0.78625201886289, 96.32687599759846, -1.85204757529826, -59.36935905485877, 47.40033549296737, -16.50741931063887, 2.28786674699375, 10260144.86]
    , [-3.030883460898826e-8, 0.00000898305509983578, 0.30071316287616, 59.74293618442277, 7.357984074871, -25.38371002664745, 13.45380521110908, -3.29883767235584, 0.32710905363475, 6856817.37]
    , [-1.981981304930552e-8, 0.000008983055099779535, 0.03278182852591, 40.31678527705744, 0.65659298677277, -4.44255534477492, 0.85341911805263, 0.12923347998204, -0.04625736007561, 4482777.06]
    , [3.09191371068437e-9, 0.000008983055096812155, 0.00006995724062, 23.10934304144901, -0.00023663490511, -0.6321817810242, -0.00663494467273, 0.03430082397953, -0.00466043876332, 2555164.4]
    , [2.890871144776878e-9, 0.000008983055095805407, -3.068298e-8, 7.47137025468032, -0.00000353937994, -0.02145144861037, -0.00001234426596, 0.00010322952773, -0.00000323890364, 826088.5]]
LL2MC = [[-0.0015702102444, 111320.7020616939, 1704480524535203.0, -10338987376042340.0, 26112667856603880, -35149669176653700, 26595700718403920, -10725012454188240, 1800819912950474, 82.5]
    , [0.0008277824516172526, 111320.7020463578, 647795574.6671607, -4082003173.641316, 10774905663.51142, -15171875531.51559, 12053065338.62167, -5124939663.577472, 913311935.9512032, 67.5]
    , [0.00337398766765, 111320.7020202162, 4481351.045890365, -23393751.19931662, 79682215.47186455, -115964993.2797253, 97236711.15602145, -43661946.33752821, 8477230.501135234, 52.5]
    , [0.00220636496208, 111320.7020209128, 51751.86112841131, 3796837.749470245, 992013.7397791013, -1221952.21711287, 1340652.697009075, -620943.6990984312, 144416.9293806241, 37.5]
    , [-0.0003441963504368392, 111320.7020576856, 278.2353980772752, 2485758.690035394, 6070.750963243378, 54821.18345352118, 9540.606633304236, -2710.55326746645, 1405.483844121726, 22.5]
    , [-0.0003218135878613132, 111320.7020701615, 0.00369383431289, 823725.6402795718, 0.46104986909093, 2351.343141331292, 1.58060784298199, 8.77738589078284, 0.37238884252424, 7.45]]


# 百度摩卡托转经纬度
def convertMC2LL(x, y):
    cF = None
    x = abs(x)
    y = abs(y)
    for cE in range(len(MCBAND)):
        if y >= MCBAND[cE]:
            cF = MC2LL[cE]
            break
    return converter(x, y, cF)


# 百度经纬度转摩卡托
def convertLL2MC(lng, lat):
    cE = None
    lng = getLoop(lng, -180, 180)
    lat = getRange(lat, -74, 74)
    for i in range(len(LLBAND)):
        if lat >= LLBAND[i]:
            cE = LL2MC[i]
            break
    if cE is not None:
        i = len(LLBAND) - 1
        while i >= 0:
            if lat <= -LLBAND[i]:
                cE = LL2MC[i]
                break
            i -= 1
    return converter(lng, lat, cE)


def converter(x, y, cE):
    xTemp = cE[0] + cE[1] * abs(x)
    cC = abs(y) /cE[9]
    yTemp = cE[2] + cE[3] * cC + cE[4] * cC * cC + cE[5] * cC * cC * cC + cE[6] * cC * cC * cC * cC + cE[7] * cC * cC * cC * cC * cC + cE[8] * cC * cC * cC * cC * cC * cC
    xTemp *= 1 if x > 0 else -1
    yTemp *= 1 if y > 0 else -1
    return [xTemp, yTemp]


def getLoop(lng, min, max):
    while lng > max:
        lng -= max - min
    while lng < min:
        lng += max - min
    return lng


def getRange(lat, min, max):
    if min:
        lat = lat if lat > min else min
    if max:
        lat = lat if lat < max else max
    return lat


# 火星坐标系到百度墨卡托
# return：[lon, lat]
def transformM2BM(point):
    bd_ll = marsTobaidu(point)
    bd_mc = convertLL2MC(bd_ll[0], bd_ll[1])
    return bd_mc


# 百度墨卡托到火星坐标系
# return：[lon, lat]
def transformBM2M(x, y):
    bd_ll = convertMC2LL(x, y)
    mars_ll = baiduTomars(bd_ll)
    return mars_ll

#
# out = [114.267833,30.563381]
# bd_3 = marsTobaidu(out)
# print(bd_3)
#
# #test =[114.274306,30.569419]
# testMC = convertLL2MC(bd_3[0],bd_3[1])
# testLL = convertMC2LL(testMC[0], testMC[1])
# print(testMC)
# print(testLL)


