# ! pip install PyKakao
from PyKakao import KakaoLocal
key = 'e6ca9e5c72e0aac012040fc4d8b512bc'
KL = KakaoLocal(key)


y, x = 33.427749, 126.662335
KL.geo_coord2regioncode(x, y)

# -----------------------#
category_group_code = "AT4"                     # 관광명소
radius = 2000                            # 반경거리(m)


result = KL.search_category(category_group_code, x, y, radius)
print(result)