import numpy as np, xarray as xr, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
S="/home/sagemaker-user/mind_the_chl_gap/data/blk_store_2014_2024/"
lat=np.load(S+"lat.npy"); lon=np.load(S+"lon.npy"); ocean=np.load(S+"ocean.npy")
ds=xr.open_dataset("/tmp/claude-1000/e.nc"); print(ds.attrs.get("title"), ds.attrs.get("summary","")[:150])
la=ds.latitude.values; lo=ds.longitude.values; z=ds.altitude.values.astype(float)
print("res",np.diff(la)[:2],np.diff(lo)[:2],z.shape)
f=RegularGridInterpolator((la,lo),z)
LA,LO=np.meshgrid(lat,lon,indexing="ij")
d=-f(np.stack([LA,LO],-1)); d=np.where(ocean,d,np.nan).astype(np.float32)
np.save(S+"depth.npy",d)
v=d[ocean]; print("depth min/med/max",v.min(),np.median(v),v.max(),"frac<=0",(v<=0).mean(),"n",v.size,"nan in ocean",np.isnan(v).sum())
src="NOAA ERDDAP etopo180, 1 arc-min"
dd=np.where(d>0,d,np.nan)
plt.figure(figsize=(9,6)); plt.imshow(np.log10(dd),origin="lower",cmap="viridis",extent=[lon[0],lon[-1],lat[0],lat[-1]])
plt.colorbar(label="log10 depth (m)"); plt.xlabel("Longitude (deg E)"); plt.ylabel("Latitude (deg N)")
plt.title(f"Black Sea bathymetry on the 2 km store grid (source: {src})"); plt.savefig("analyses/spatial_node/figures/blk_depth.png",dpi=130,bbox_inches="tight"); plt.close()
names=["Danube","Dniester","Dnieper-Bug","Kerch","Rioni","Sakarya","Kizilirmak","Bosporus"]
pts=[(45.2,29.7),(46.1,30.4),(46.6,31.6),(45.3,36.6),(42.2,41.6),(41.1,30.6),(41.7,36.0),(41.2,29.1)]
def hav(a,b,c,e):
    a,b,c,e=map(np.radians,(a,b,c,e)); h=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((e-b)/2)**2
    return 2*6371.0088*np.arcsin(np.sqrt(h))
oi=np.argwhere(ocean); out=[];snap=[]
for n,(a,b) in zip(names,pts):
    dist=hav(a,b,lat[oi[:,0]],lon[oi[:,1]]); i,j=oi[dist.argmin()]
    snap.append((lat[i],lon[j])); print(n,"snap",lat[i],lon[j],"offset km",dist.min())
    out.append(hav(lat[i],lon[j],LA,LO))
R=np.stack(out).astype(np.float32); np.save(S+"river_dist.npy",R); json.dump(names,open(S+"river_names.json","w"))
print(R.shape,R.dtype)
m=np.where(ocean,R.min(0),np.nan)
plt.figure(figsize=(9,6)); plt.imshow(m,origin="lower",cmap="viridis",extent=[lon[0],lon[-1],lat[0],lat[-1]]); plt.colorbar(label="Distance to nearest river mouth (km)")
for n,(a,b) in zip(names,snap): plt.plot(b,a,"r^",ms=7,mec="w"); plt.text(b+.15,a+.1,n,color="w",fontsize=8,bbox=dict(fc="k",alpha=.5,pad=1))
plt.xlabel("Longitude (deg E)"); plt.ylabel("Latitude (deg N)"); plt.title("Minimum great-circle distance to the 8 river mouths (km), mouths marked")
plt.savefig("analyses/spatial_node/figures/blk_river_dist.png",dpi=130,bbox_inches="tight")
