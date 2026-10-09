"""Rebuild figures, numerical tables and checks from saved statistics; no random draws."""
from pathlib import Path
import os, json, csv, hashlib, re, sys, platform, tempfile, argparse
sys.dont_write_bytecode = True
def require_matplotlib_panel_alignment(fig, *, strict, require_panel_labels,
                                       tolerance_pt, json_out=None):
    """Check the equal-size row panels used by this repository's four figures."""
    width_pt, height_pt = fig.get_size_inches() * 72
    boxes = np.array([[a.get_position().x0 * width_pt,
                       a.get_position().y0 * height_pt,
                       a.get_position().width * width_pt,
                       a.get_position().height * height_pt] for a in fig.axes])
    deviations = {}
    if len(boxes) > 1:
        for key, column in [('bottom', 1), ('width', 2), ('height', 3)]:
            deviations[key] = float(np.ptp(boxes[:, column]))
        deviations['top'] = float(np.ptp(boxes[:, 1] + boxes[:, 3]))
    if len(boxes) > 2:
        gaps = boxes[1:, 0] - (boxes[:-1, 0] + boxes[:-1, 2])
        deviations['gutter'] = float(np.ptp(gaps))
    if require_panel_labels:
        anchors = []
        for index, axis in enumerate(fig.axes):
            label = next((t for t in axis.texts if t.get_text() == chr(97 + index)), None)
            if label is None:
                raise ValueError(f'Missing panel label {chr(97 + index)}')
            pixel = label.get_transform().transform(label.get_position())
            anchors.append(fig.transFigure.inverted().transform(pixel) * [width_pt, height_pt])
        deviations['label_height'] = float(np.ptp(np.asarray(anchors)[:, 1]))
        deviations['label_left_offset'] = float(np.ptp(np.asarray(anchors)[:, 0] - boxes[:, 0]))
    passed = all(value <= tolerance_pt for value in deviations.values())
    report = {'passed': passed, 'panel_count': len(boxes), 'tolerance_pt': tolerance_pt,
              'maximum_deviations_pt': deviations, 'scope': 'equal-size horizontal row panels'}
    if json_out is not None:
        Path(json_out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    if strict and not passed:
        raise ValueError(f'Panel alignment failed: {report}')
    return report

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'student-t-mpl'))
import numpy as np
import scipy
from scipy import stats, special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=ROOT/'data'; F=ROOT/'figures'; F.mkdir(exist_ok=True)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--summaries-dir', type=Path, help='Optional output directory for numerical CSV summaries')
parser.add_argument('--manuscript-dir', type=Path, help='Optional private manuscript directory for exact numerical table checks')
parser.add_argument('--audit-dir', type=Path, help='Optional output directory for figure alignment reports')
parser.add_argument('--replication-table-only', action='store_true',
                    help='Export the full replicated table to --summaries-dir, without drawing figures or editing TeX')
args=parser.parse_args()
MANUSCRIPT_DIR = args.manuscript_dir.resolve() if args.manuscript_dir else None
checked_manuscript_tables = []
if MANUSCRIPT_DIR is not None:
    for filename in ('parameter estimate.tex', 'supplement.tex'):
        if not (MANUSCRIPT_DIR / filename).is_file():
            parser.error(f'Missing manuscript file: {MANUSCRIPT_DIR / filename}')
for directory in [args.summaries_dir,args.audit_dir]:
    if directory: directory.mkdir(parents=True,exist_ok=True)
cfg=json.loads((D/'config.json').read_text())
MU,THETA,ALPHA=cfg['mu'],cfg['theta'],cfg['alpha']
COL=['#0072B2','#D55E00','#009E73']
# Physical width matches the manuscript text width; all PDFs remain vector.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,
 'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,
 'axes.linewidth':.7,'lines.linewidth':1.25,'lines.markersize':4,
 'xtick.labelsize':8,'ytick.labelsize':8,'figure.dpi':120})

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(D/'simulation_arrays.npz') == '9ec146f3e14b30b6635af9455d9ef2304ed918dadf66636f753f213eac49801a', 'Saved data checksum mismatch'
archive=np.load(D/'simulation_arrays.npz')
def load_saved(name):
    prefix=Path(name).stem+'/'
    return {k[len(prefix):]:archive[k] for k in archive.files if k.startswith(prefix)}

def savefig(fig,name):
    # Separate bold panel letters from titles to make final alignment measurable.
    for i,a in enumerate(fig.axes):
        title=re.sub(r'^\([a-z]\)\s*','',a.get_title())
        a.set_title(title,pad=10)
        a.text(0,1.075,chr(97+i),transform=a.transAxes,
               fontweight='bold',fontsize=9,ha='left',va='bottom')
        if name != 'recovery': a.grid(axis='y',color='0.90',lw=.5,zorder=0)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,strict=True,require_panel_labels=True,
        tolerance_pt=1.5,json_out=(args.audit_dir/(name+'_alignment.json') if args.audit_dir else None))
    fig.savefig(F/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

def se(x): return float(np.std(x,ddof=1)/np.sqrt(len(x)))
def logmetrics(est):
    assert np.all(np.isfinite(est)) and np.all(est > 0)
    le=np.log(est/THETA); loss=le**2; lr=float(np.sqrt(loss.mean()))
    return lr,se(loss)/(2*lr)
def put(path,name,content):
    """Optionally verify private manuscript table cells; never edit TeX."""
    if MANUSCRIPT_DIR is None:
        return
    if r'\begin{table}' not in content and r'\begin{longtable}' not in content: return
    existing=path.read_text(encoding='utf-8')
    match=re.search(r'% BEGIN '+name+r'\n(.*?)% END '+name,existing,flags=re.S)
    assert match, (path,name)
    def cells(text):
        return [line.strip() for line in text.splitlines() if re.match(r'^\d+ & ',line)]
    assert cells(match[1])==cells(content), f'Table values changed: {name}'
    checked_manuscript_tables.append(name)

def table(caption,label,cols,head,rows):
    return ('\\begin{table}[H]\n\\centering\n\\caption{'+caption+'}\\label{'+label+'}\n'
      '\\begin{tabular}{'+cols+'}\\toprule\n'+head+'\\\\\\midrule\n'
      +'\n'.join(rows)+'\n\\bottomrule\\end{tabular}\n\\end{table}')
def csvout(name,rows):
    if args.summaries_dir is None: return
    with (args.summaries_dir/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def replication_grid_rows():
    """All 45 saved (m, nu, J) settings; no simulation or refitting."""
    result=[]
    for m in cfg['replicated']['m']:
        for nu in cfg['replicated']['nu']:
            zz=load_saved(f'replicated_m{m}_nu{nu}_estimates.npz')
            for J in cfg['replicated']['J']:
                est=zz[f'J{J}_residual_likelihood']
                assert len(est)==cfg['replicated']['replications']
                lr,lrse=logmetrics(est);le=np.log(est/THETA);vv=2*(m+nu+2)/(m*nu)
                p=float(np.mean(abs(le)<=stats.norm.ppf(1-ALPHA/2)*np.sqrt(vv/J)))
                result.append(dict(m=m,nu=nu,J=J,valid=len(est),log_rmse=lr,
                    coverage=p,coverage_mcse=float(np.sqrt(p*(1-p)/len(est)))))
    return result

def replication_longtable(result):
    caption=(r'Residual likelihood over all saved \((m,\nu,J)\) settings, '
             r'\(M=5000\). Coverage uses the asymptotic scale interval; MCSE denotes '
             r'the Monte Carlo standard error of the coverage estimate.')
    head=r'\(m\)&\(\nu\)&\(J\)&Log RMSE&Coverage&Coverage MCSE'
    rows=[f"{r['m']} & {r['nu']} & {r['J']} & {r['log_rmse']:.4f} & {r['coverage']:.4f} & {r['coverage_mcse']:.4f}"+r'\\'
          for r in result]
    return ('\\begin{longtable}{rrrrrr}\n\\caption{'+caption+'}\\label{tab:repall}\\\\\n'
            '\\toprule\n'+head+'\\\\\\midrule\n\\endfirsthead\n'
            '\\multicolumn{6}{c}{Table~\\thetable\\ (continued)}\\\\\n'
            '\\toprule\n'+head+'\\\\\\midrule\n\\endhead\n'
            '\\midrule\\multicolumn{6}{r}{Continued on next page}\\\\\n\\endfoot\n'
            '\\bottomrule\n\\endlastfoot\n'+'\n'.join(rows)+'\n\\end{longtable}')

def export_replication_table(result,content):
    csvout('replication_all.csv',result)
    if args.summaries_dir:
        (args.summaries_dir/'replication_all.tex').write_text(content+'\n',encoding='utf-8')

if args.replication_table_only:
    if args.summaries_dir is None: parser.error('--replication-table-only requires --summaries-dir')
    full_grid=replication_grid_rows()
    export_replication_table(full_grid,replication_longtable(full_grid))
    print(json.dumps({'new_random_draws':0,'replicated_settings_exported':len(full_grid)},indent=2))
    raise SystemExit(0)

single=[];sizes=cfg['brownian']['sizes'];nu=10
arrays={s:load_saved(f'brownian_{s}_statistics.npz') for s in ['fixed','long']}
for design,z in arrays.items():
    for nv in cfg['brownian']['nu']:
        for n in sizes:
            mu,q,w=z[f'nu{nv}_n{n}'];m=n-1
            a=1 if design=='fixed' else .01*n
            assert len(q)==6000 and (w>0).all() and (q>0).all()
            dm=(mu-MU)**2; wm=(q/m-w)**2;cm=((nv-2)/nv*q/m-THETA)**2
            single.append(dict(design=design,nu=nv,n=n,m=m,a=a,valid=len(q),
             drift_mse=float(dm.mean()),drift_mcse=se(dm),W_mse=float(wm.mean()),
             W_mcse=se(wm) if nv>8 else None,D_mse=float(cm.mean()),
             W_theory=2*THETA**2*nv**2/(m*(nv-2)*(nv-4)),
             D_theory=2*THETA**2/(nv-4)*(1+(nv-2)/m),
             drift_theory=THETA*nv/((nv-2)*a),
             relative_W_mse=float(np.mean((q/m/w-1)**2)),
             mle_limit_mse=float(np.mean((q/n-w)**2))))
csvout('single_summary.csv',single)
fig,ax=plt.subplots(1,2,figsize=(6.4,3.25))
fig.subplots_adjust(left=.115,right=.98,bottom=.19,top=.82,wspace=.47)
row=[r for r in single if r['nu']==10 and r['design']=='long']
x=np.array([r['a'] for r in row])
xx=np.geomspace(min(x)*.85,max(x)*1.2,150)
ax[0].plot(xx,THETA*10/8/xx,color=COL[0],lw=1.4,label='Exact risk',ls='--')
ax[0].errorbar(x,[r['drift_mse'] for r in row],yerr=[1.96*r['drift_mcse'] for r in row],
 fmt='o',color=COL[0],capsize=3,label='Growing horizon')
r=next(r for r in single if r['nu']==10 and r['design']=='fixed' and r['n']==1024)
ax[0].errorbar([1],[r['drift_mse']],yerr=[1.96*r['drift_mcse']],fmt='s',
 color=COL[1],capsize=3,label='Fixed interval (all grids)')
ax[0].set(xscale='log',yscale='log',xlabel=r'Drift information factor $a_n$',ylabel=r'MSE of $\widehat\mu_n$',title='(a) Drift estimation')
ax[0].legend(frameon=False,fontsize=7,loc='lower left',handlelength=1.7)
row=[r for r in single if r['nu']==10 and r['design']=='fixed']
ax[1].plot([r['m'] for r in row],[r['W_theory'] for r in row],color=COL[2],label='Exact risk',ls='--')
ax[1].errorbar([r['m'] for r in row],[r['W_mse'] for r in row],yerr=[1.96*r['W_mcse'] for r in row],
 fmt='o',color=COL[2],capsize=3,label='Fixed interval')
ax[1].set(xscale='log',yscale='log',xlabel=r'Residual dimension $m=n-1$',ylabel=r'MSE of $\widetilde W_n$',title='(b) Realized scale')
ax[1].legend(frameon=False,fontsize=7,loc='upper right');savefig(fig,'recovery')

fig,ax=plt.subplots(1,2,figsize=(6.4,3.10),sharey=True)
fig.subplots_adjust(left=.115,right=.98,bottom=.19,top=.82,wspace=.14)
qrows=[]
for i in range(2):
    samples=[]
    for n in sizes:
        mu,q,w=arrays['fixed'][f'nu10_n{n}']
        est=q/n if i==0 else .8*q/(n-1)
        samples.append(est-THETA)
    samples.append((w if i==0 else .8*w)-THETA)
    boxes=[]
    for tag,s in zip([str(n) for n in sizes]+['Limit'],samples):
        qs=np.quantile(s,[.05,.25,.5,.75,.95])
        boxes.append(dict(label=tag,whislo=qs[0],q1=qs[1],med=qs[2],q3=qs[3],whishi=qs[4],fliers=[]))
        qrows.append(dict(estimator='mle' if i==0 else 'D',n=tag,q05=qs[0],q25=qs[1],q50=qs[2],q75=qs[3],q95=qs[4],valid=len(s)))
    art=ax[i].bxp(boxes,showfliers=False,patch_artist=True,
     boxprops={'facecolor':COL[i],'alpha':.28},medianprops={'color':'black'})
    art['boxes'][-1].set_facecolor('#888888')
    art['boxes'][-1].set_linestyle('--')
    ax[i].axhline(0,color='0.5',lw=.8,ls='--')
    ax[i].set(xlabel=r'Observations $n$ and latent limit',title=['(a) Marginal scale MLE','(b) Bias-corrected estimate'][i])
ax[0].set_ylabel(r'Error relative to fixed $\theta$')
savefig(fig,'random_limits');csvout('random_limit_quantiles.csv',qrows)

rep=[];qq=[];probs=np.linspace(.01,.99,99);m=7
fig,ax=plt.subplots(1,2,figsize=(6.4,3.30))
figq,aq=plt.subplots(1,3,figsize=(6.4,2.95),sharex=True,sharey=True)
fig.subplots_adjust(left=.115,right=.98,bottom=.29,top=.82,wspace=.46)
figq.subplots_adjust(left=.095,right=.985,bottom=.29,top=.81,wspace=.20)
for i,nu in enumerate([1,3,10]):
    z=load_saved(f'replicated_m7_nu{nu}_estimates.npz')
    vv=2*(m+nu+2)/(m*nu);rr=[]
    for ji,J in enumerate([10,50,200]):
        est=z[f'J{J}_residual_likelihood']
        assert len(est)==5000 and (est>0).all() and np.isfinite(est).all()
        lr,lrse=logmetrics(est);le=np.log(est/THETA)
        coverage=float(np.mean(abs(le)<=stats.norm.ppf(.975)*np.sqrt(vv/J)))
        Z=np.sqrt(J)*(est-THETA)/(THETA*np.sqrt(vv))
        ez=np.quantile(Z,probs);dist=float(stats.kstest(Z,'norm').statistic)
        for p,t,e in zip(probs,stats.norm.ppf(probs),ez):
            qq.append(dict(nu=nu,m=m,J=J,p=p,normal_quantile=t,empirical_quantile=e))
        row=dict(nu=nu,m=m,J=J,valid=len(est),log_rmse=lr,log_rmse_mcse=lrse,
         coverage=coverage,coverage_mcse=float(np.sqrt(coverage*(1-coverage)/len(est))),
         log_bias=float(le.mean()),log_bias_mcse=se(le),normal_cdf_distance=dist,
         q025=float(np.quantile(Z,.025)),q50=float(np.median(Z)),q975=float(np.quantile(Z,.975)),
         bias=float(np.mean(est-THETA)) if nu==10 else None,
         bias_mcse=se(est-THETA) if nu==10 else None,
         rmse=float(np.sqrt(np.mean((est-THETA)**2))) if nu==10 else None)
        if nu==10: row['rmse_mcse']=se((est-THETA)**2)/(2*row['rmse'])
        else:row['rmse_mcse']=None
        rep.append(row);rr.append(row)
        aq[i].plot(stats.norm.ppf(probs),ez,color=COL[ji],ls=[':', '--','-'][ji],lw=1.6,label=f'J = {J}')
    js=np.array([r['J'] for r in rr])
    ax[0].errorbar(js,[r['log_rmse'] for r in rr],yerr=[1.96*r['log_rmse_mcse'] for r in rr],fmt=['o-','s-','^-'][i],color=COL[i],capsize=3,label=rf'$\nu={nu}$')
    ax[0].plot(js,np.sqrt(vv/js),color=COL[i],ls='--',alpha=.7,lw=1)
    # Horizontal offsets separate MC intervals; ticks continue to denote J.
    ax[1].errorbar(js*(1+(i-1)*.06),[r['coverage'] for r in rr],
     yerr=[1.96*r['coverage_mcse'] for r in rr],fmt=['o-','s-','^-'][i],color=COL[i],capsize=3,label=rf'$\nu={nu}$')
    aq[i].plot([-2.5,2.5],[-2.5,2.5],color='0.4',lw=.8,ls='--',zorder=0)
    aq[i].set(title=rf'$\nu={nu}$',xlabel='Normal quantile')
    aq[i].set_xticks([-2,0,2]); aq[i].set_yticks([-2,0,2,4])
    aq[i].set_xlim(-2.65,2.65)
aq[0].set_ylabel(r'Empirical quantile of $Z_J$')
figq.legend(*aq[0].get_legend_handles_labels(),frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,.005))
savefig(figq,'replication_normal')
for a in ax:
    a.set_xscale('log');a.set_xticks([10,50,200],['10','50','200']);a.set_xlabel(r'Independent trajectories $J$')
ax[0].set(yscale='log',ylabel='Log RMSE',title='(a) Fixed-scale error')
ax[1].axhline(.95,color='0.4',ls='--',lw=1)
ax[1].set(ylabel='Coverage',title='(b) Interval coverage')
fig.legend(*ax[1].get_legend_handles_labels(),frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,.005))
ax[1].set_ylim(.927,.962)
savefig(fig,'replication')
csvout('replication_summary.csv',rep);csvout('normal_quantiles.csv',qq)
main=(MANUSCRIPT_DIR or ROOT)/'parameter estimate.tex';supp=(MANUSCRIPT_DIR or ROOT)/'supplement.tex'
rows=[]
for r in rep:
    bias=f"{r['bias']:.4f} ({r['bias_mcse']:.4f})" if r['bias'] is not None else '--'
    rmse=f"{r['rmse']:.4f}" if r['rmse'] is not None else '--'
    rows.append(f"{r['nu']} & {r['J']} & {r['log_rmse']:.4f} & {r['coverage']:.4f} & {bias} & {rmse}"+r'\\')
put(main,'REPLICATION TABLE',table(
 r'Residual likelihood at fixed \(m=7\), \(M=5000\). Bias and raw RMSE concern \(\theta\) and are reported at \(\nu=10\); parentheses give the bias Monte Carlo standard error. Dashes denote unreported raw-moment summaries, not failed fits. Log RMSE and coverage use all tail settings.',
 'tab:rep','rrrrrr',r'\(\nu\)&\(J\)&Log RMSE&Coverage&Bias (MCSE)&Raw RMSE',rows))
dist=lambda nu,J:next(r['normal_cdf_distance'] for r in rep if r['nu']==nu and r['J']==J)
findings=(r'At \(\nu=1\), the empirical distribution-function distance to the standard normal falls from '
f"{dist(1,10):.4f}"+r' at \(J=10\) to '+f"{dist(1,200):.4f}"+r' at \(J=200\). '
r'The quantile curves retain right-tail departures at small \(J\), particularly under heavier mixing tails. '
r'These finite-sample discrepancies coexist with decreasing estimation error and do not make the asymptotic interval exact. '
r'The complete \((m,\nu,J)\) grid, including the small-\(J\) coverage departures, is retained in Supplement~\ref{supp:coverage}.')
put(main,'REPLICATION FINDINGS',findings)

coverage=[dict(nu=nu,n=n) for nu in cfg['coverage']['nu'] for n in cfg['coverage']['sizes']]
cz=load_saved('coverage_raw.npz');cr=[]
for r in coverage:
    nu,n=r['nu'],r['n'];w,A,U=cz[f'nu{nu}_n{n}'];q=w*U;dm=n-1
    mulen=2*stats.t.ppf(.975,dm)*np.sqrt(q/dm)
    wlen=q*(1/stats.chi2.ppf(.025,dm)-1/stats.chi2.ppf(.975,dm))
    tlen=q/dm*(1/stats.f.ppf(.025,dm,nu)-1/stats.f.ppf(.975,dm,nu))
    ph=[float(np.mean(abs(A)/np.sqrt(U/dm)<=stats.t.ppf(.975,dm))),
        float(np.mean((U>=stats.chi2.ppf(.025,dm))&(U<=stats.chi2.ppf(.975,dm)))),
        float(np.mean((q/(dm*THETA)>=stats.f.ppf(.025,dm,nu))&(q/(dm*THETA)<=stats.f.ppf(.975,dm,nu))))]
    lens=[float(np.median(x)) for x in [mulen,wlen,tlen]]
    cr.append(f"{nu} & {n} & "+' & '.join(f'{x:.4f}' for x in ph)+' & '+' & '.join(f'{x:.3f}' for x in lens)+r'\\')
put(supp,'COVERAGE TABLE',table(
 r"Exact 95\% interval coverage and median widths, \(M=12000\), \(a_n=1\). Widths use each target's original units; nominal coverage MCSE is \(0.0020\).",
 'tab:covall','rr rrr rrr',r'&&\multicolumn{3}{c}{Coverage}&\multicolumn{3}{c}{Median width}\\'
 r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\(\nu\)&\(n\)&\(C_\mu\)&\(C_W\)&\(C_\theta\)&\(C_\mu\)&\(C_W\)&\(C_\theta\)',cr))
allrep=replication_grid_rows()
replication_table=replication_longtable(allrep)
export_replication_table(allrep,replication_table)
put(supp,'SENSITIVITY TABLE',replication_table)
rows=[f"{r['nu']} & {r['n']} & {r['W_mse']:.4f} & {r['W_theory']:.4f} & {r['D_mse']:.4f} & {r['D_theory']:.4f}"+r'\\'
 for r in single if r['design']=='fixed']
put(supp,'TAIL TABLE',table(
 r'Fixed-interval squared scale errors for every saved tail setting, \(M=6000\). At \(\nu=5\), squared losses have infinite variance; their empirical means can be unstable. All observations are retained.',
 'tab:tails','rrrrrr',r'\(\nu\)&\(n\)&MSE \(\widetilde W_n\)&Theory&MSE \(D_n\)&Theory',rows))
budget=[]
for n in cfg['design']['sizes']:
    J=cfg['design']['budget']//n
    est=load_saved(f'design_n{n}_estimates.npz')[f'J{J}_residual_likelihood']
    lr,lse=logmetrics(est)
    budget.append(dict(n=n,J=J,valid=len(est),log_rmse=lr,log_rmse_mcse=lse,
       residual_information=J*(n-1)*10/(2*THETA**2*(n+11))))
csvout('allocation.csv',budget)
b4=next(r for r in budget if r['n']==4);b5=next(r for r in budget if r['n']==5)
diff=b4['log_rmse']-b5['log_rmse'];dse=np.hypot(b4['log_rmse_mcse'],b5['log_rmse_mcse'])
put(supp,'ALLOCATION FINDINGS',
 r'The continuous optimum is \(1+\sqrt{12}\); \(n=4,5\) tie under both the cost ratio and this budget. Their log RMSEs are '
f"{b4['log_rmse']:.5f} and {b5['log_rmse']:.5f}"+r', respectively. The difference is '
f"{diff:.6f}"+r', with Monte Carlo standard error '+f"{dse:.6f}"+
r' from independent datasets across the two designs. The simulation therefore does not resolve an ordering between these tied allocations. The full budget grid is saved in the numerical data.')
report={'new_random_draws':0,'saved_arrays_verified':len(archive.files),'replicated_settings_checked':len(allrep),
 'interval_settings_checked':len(coverage)*3,
 'raw_moment_reporting_nu':10,'failed_or_removed_main_datasets':0,
 'manuscript_numeric_tables_verified':checked_manuscript_tables,
 'manuscript_checks_enabled':MANUSCRIPT_DIR is not None,
 'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform()},
 'figure_sha256':{p.name:sha(p) for p in F.glob('*.pdf')}}
if args.audit_dir: (args.audit_dir/'analysis_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
