# Literature notes: physics-informed NN for cloud-gap CHL in the western Indian Ocean

Compiled 2026-09-30 from web/literature search (delegated search agent, then checked by hand where possible).
Verification key: [V] primary source read; [S] taken from search snippets, page not fetched (paywall/403); [U] textbook-level or own arithmetic, not verified against a source.

## 1. Governing equations and upwelling physics

Surface tracer equation (mixed-layer averaged over MLD h) [U, textbook]:

    dC/dt + u dC/dx + v dC/dy = div(K_H grad C) + S_w + R(C)

S_w = vertical exchange, e.g. max(w_E + dh/dt, 0) (C_sub - C)/h ; R(C) = net biological rate. "Trusted transport, learned biology" is the split used by PPIDM and PeRCNN (below).

Wind stress: tau = rho_air C_D |U10| U10, rho_air ~ 1.22 kg/m3. Large & Pond (1981): C_D = 1.2e-3 for 4 <= |U10| < 11 m/s; (0.49 + 0.065 |U10|) e-3 for 11-25 m/s [S: Jacox et al. 2018 https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JC014187 ; https://ams.confex.com/ams/pdfpapers/86740.pdf].

Ekman transport M = (tau x k)/f (kg/m/s), M_x = tau_y/f, M_y = -tau_x/f. Ekman pumping
    w_E = curl_z( tau / (rho_w f) ) = (1/(rho_w f)) (dtau_y/dx - dtau_x/dy) + beta tau_x /(rho_w f^2)
Stewart (2008) Introduction to Physical Oceanography ch. 9.4, 12.4 (CC BY-NC-SA) [V]: https://geo.libretexts.org/Bookshelves/Oceanography/Introduction_to_Physical_Oceanography_(Stewart)/12:_Vorticity_in_the_Ocean/12.4:_Vorticity_and_Ekman_Pumping  (sign convention differs; positive = upward here).

Bakun coastal upwelling index: offshore Ekman volume transport per 100 m coastline = tau_alongshore/(rho_w f) [S: Jacox et al. 2018; Schwing et al. 1996 NOAA Tech. Memo http://www.mjacox.com/wp-content/uploads/2018/06/Schwing_etal_UpwellingIndices_1996.pdf].
Worked number [U]: 15 m/s wind -> C_D 1.46e-3 -> tau 0.40 Pa; at 10N (f = 2.5e-5) M/rho_w ~ 15 m2/s ~ 1500 m3/s per 100 m coast.

Magnitudes, Arabian Sea / Somali system:
- Chatterjee et al. (2019) Sci. Rep. "Annihilation of the Somali upwelling system during summer monsoon" https://pmc.ncbi.nlm.nih.gov/articles/PMC6527695/ [V]: Ekman pumping > 4 m/day (~4.6e-5 m/s) north Somali section at peak monsoon; offshore Ekman transport ~1e4 kg/m/s; Great Whirl currents to 250 cm/s; upwelling peaks May-July and largely shuts down by August despite favourable winds (thermocline deepens 20 -> 80 m). Wind -> upwelling is not monotone here.
- Vinayachandran et al. (2021) Biogeosciences 18, 5967 https://bg.copernicus.org/articles/18/5967/2021/ [V]: Somali currents 250-300 cm/s; upwelled nitrate > 15 uM.
- J. Operational Oceanography (2023) https://www.tandfonline.com/doi/full/10.1080/1755876X.2023.2186588 [S]: JJAS Ekman pumping > 1 m/day off northern Yemen; SW-monsoon CHL 50-200 % above NE-monsoon; alongshore transport dominates curl-driven pumping.
- Beal et al. (2013) JPO https://journals.ametsoc.org/view/journals/phoc/43/9/jpo-d-13-033.1.xml [S]: negative curl (downwelling) offshore of the Findlater jet axis, positive curl (upwelling) inshore.
- SE Arabian Sea upwelling velocities 0.5-1.5 m/day [S: https://www.sciencedirect.com/science/article/abs/pii/S2352485520306472].

CHL response lags: fronts lead CHL by ~1 month (Wang et al. 2021 J. Mar. Syst. https://www.sciencedirect.com/science/article/pii/S0924796321000439 [S]); Gulf of Aden CHL peaks 1-2 months after thermocline shoaling (Yao & Hoteit 2015 PLoS One https://pmc.ncbi.nlm.nih.gov/articles/PMC4366251/ [V]); event scale 3-4 days (cyclone review https://arxiv.org/pdf/2012.04384 [S]); offshore blooms lag open-sea upwelling > 1 month (https://link.springer.com/article/10.1007/s12040-020-01512-y [S]).

Equatorial singularity: M and w_E scale as 1/f, 1/f^2; invalid within ~2-3 deg of the equator (Stewart 14.1 https://geo.libretexts.org/Bookshelves/Oceanography/Introduction_to_Physical_Oceanography_(Stewart)/14:_Equatorial_Processes/14.1:_Equatorial_Processes [S]). Domain reaches 5S, so taper/mask the Ekman terms.

## 2. Prior work

Foundational: Raissi, Perdikaris, Karniadakis (2019) JCP 378, 686 [S]; Raissi, Yazdani, Karniadakis (2020) Hidden fluid mechanics, Science 367, 1026 https://github.com/maziarraissi/HFM [S]; Karniadakis et al. (2021) Nat. Rev. Phys. https://www.nature.com/articles/s42254-021-00314-5 [S].

Gap-filling baselines: DINEOF (Beckers & Rixen 2003 JAOT; Alvera-Azcarate et al. 2005) [S]; DINEOF global ocean colour https://journals.ametsoc.org/view/journals/atot/41/10/JTECH-D-23-0105.1.xml [S]; DINCAE 1.0 Barth et al. (2020) GMD 13, 1609 https://gmd.copernicus.org/articles/13/1609/2020/ [S]; DINCAE on CHL South China Sea, Han et al. (2020) Remote Sens. 12, 480 https://doi.org/10.3390/rs12030480 [S]; Krasnopolsky et al. (2016) NN gap filling of ocean colour https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4706868/ [S]; OCNET Hong et al. (2023) ESSD 15, 5281 https://essd.copernicus.org/articles/15/5281/2023/ [V]: U-Net, inputs SST, salinity, PAR, pressure, CHL climatology, 0.25 deg daily 2001-2021, CC 0.6-0.7; 4DVarNet-SSH Fablet et al. (2023) GMD 16, 2119 https://gmd.copernicus.org/articles/16/2119/2023/ [S]; 4DVarNet ocean colour Nguyen, Dorffer, Jourdin, Fablet (2025) IEEE JSTARS https://arxiv.org/abs/2503.11588 [V] (> 20 % RMSLE gain over OI/DINEOF per https://arxiv.org/html/2503.11532 [S]).

Physics-constrained CHL / tracers: Xu & Li (2025) "Partial Physics Informed Diffusion Model for Ocean Chlorophyll Concentration Reconstruction", NeurIPS 2025 https://openreview.net/pdf?id=B5fFInNlhZ [V abstract]: enforces only the trusted advection operator, penalises the residual discrepancy between prediction and truth, learns growth/loss. Sammartino et al. (2025) Env. Model. Softw. https://doi.org/10.1016/j.envsoft.2025.106660 [S]: CNN-LSTM, physics loss on density consistency. SSTODE (AAAI 2025) https://arxiv.org/pdf/2511.05629 [S]: neural ODE with explicit advection-diffusion for SST. PDE-guided NN for SST, Remote Sens. 15, 3498 https://www.mdpi.com/2072-4292/15/14/3498 [S]. PINN 2-D advection-diffusion pollutant at sea https://arxiv.org/abs/2507.08834 [S]. Optimally weighted PINNs in ocean modelling https://arxiv.org/abs/2106.08747 [S]. Hybrid BGC: Neural-BGC Ouala et al. (2026) GRL https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2026GL123123 [S]; neural closure models Gupta & Lermusiaux https://arxiv.org/pdf/2012.13869 [S].
No paper found that uses Ekman pumping / wind-stress curl as a PINN residual term for CHL; Ekman quantities appear as features only.

## 3. PINN training pitfalls and remedies

- Krishnapriyan et al. (2021) NeurIPS https://arxiv.org/abs/2109.01050 [V]: PINNs fail on convection, reaction, reaction-diffusion; remedies curriculum regularisation and sequence-to-sequence (short windows).
- Wang, Teng, Perdikaris (2021) SIAM J. Sci. Comput. 43(5) https://dl.acm.org/doi/10.1137/20M1318043 [S]: gradient imbalance; learning-rate-annealed loss weights.
- Wang, Yu, Perdikaris (2022) JCP 449, 110768 [S]: NTK-based adaptive weights.
- Wang, Wang, Perdikaris (2021) CMAME 384 https://www.sciencedirect.com/science/article/abs/pii/S0045782521002759 [S]: spectral bias; multi-scale Fourier features.
- Wang, Sankaran, Perdikaris (2022/2024) "Respecting causality" https://arxiv.org/abs/2203.07404 [S]: causal weights w_i = exp(-eps sum_{k<i} L_k).
- Mojgani, Balajewicz, Hassanzadeh (2023) Lagrangian PINNs CMAME 404, 115810 https://github.com/rmojgani/LPINNs [S]: convection-dominated problems are the hard case; NTK for advection-diffusion https://arxiv.org/pdf/2211.11716 [S].
- Lu et al. (2021) hard constraints https://arxiv.org/abs/2102.04626 [S].
- Gridded / finite-difference residuals: Zhu et al. (2019) JCP 394, 56 https://github.com/cics-nd/pde-surrogate [S]; PhyGeoNet Gao, Sun, Wang (2021) JCP 428 https://github.com/Jianxun-Wang/phygeonet [S]; PhyCRNet Ren et al. (2022) CMAME 389 https://arxiv.org/abs/2106.14103 [S]; PeRCNN Rao et al. (2023) Nat. Mach. Intell. https://www.nature.com/articles/s42256-023-00685-7 [S]: known PDE terms hard-wired as conv layers, unknown reaction term learned.

## 4. Practical scaling at 0.25 deg / daily  [U, arithmetic from cited magnitudes]

- Grid 0.25 deg ~ 27.8 km meridional; 26.8 km zonal at 15N.
- Courant number u dt/dx, dt = 1 day: u 0.3 -> 0.95; 0.5 -> 1.6; 1 -> 3.2; 2.5 -> 8. Explicit daily residual violates CFL in boundary currents; semi-Lagrangian or sub-cycled residual needed.
- Eddy diffusivity at ~40 km: ~3000 m2/s tropics / western boundary currents, ~100 m2/s gyres (Groeskamp et al. 2021 JPO https://journals.ametsoc.org/view/journals/phoc/51/2/jpo-d-19-0256.1.xml [S]; Okubo 1971 https://falk.ucsd.edu/reading/okubo1971.pdf). Grid Peclet 13-135: diffusion is a regulariser, not a constraint.
- w_E 1 m/day over h = 30 m -> exchange rate 4e-7 /s (30 d e-folding); 4 m/day -> 8 d. Biological rates 0.1-1 /day dominate except at the coastal extreme.

## 5. Openly licensed schematics

- Coastal upwelling, public domain (Lichtspiel), Southern-Hemisphere drawing: https://commons.wikimedia.org/wiki/File:Upwelling-labels-en.svg [V]  -> figures/upwelling_wikimedia_pd.svg
- Ekman spiral/transport, CC BY-SA 4.0 (Dutchoceanographer): https://upload.wikimedia.org/wikipedia/commons/b/bd/Sketch_ocean_zonal_wind_stress_ekman_spiral_transport.jpg [V] -> figures/ekman_spiral_ccbysa.jpg
- NOAA National Ocean Service upwelling graphic (US Government work): https://oceanservice.noaa.gov/facts/upwelling.html [V] -> figures/upwelling_noaa.jpg
- No open Ekman-pumping or PINN-architecture figure found; drawn in chl_cnn/schematics.py.

## Not verified / open
Jacox et al. 2018 and the 2023 upwelling-index paper were paywalled; Okubo fit constants not extracted; PPIDM operator/dataset details not extracted.
