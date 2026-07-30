// ── game-logic.js (ausgelagert aus index.html) ─────────────
// Pure Spiellogik: BFS, Wand-Validierung, Hammer-Kontext.
// Referenziert COLS/ROWS/CELL/WW/CUR_MAP aus globalem Scope (gesetzt vom Babel-Script).
// ── Game logic ─────────────────────────────────────────────
function ck(r,c){return `${r},${c}`;}
function edgeBlocked(r,c,nr,nc,walls){
  const dr=nr-r,dc=nc-c;
  if(dr===1)  return walls.has(`H-${r}-${c}`)  ||(c>0&&walls.has(`H-${r}-${c-1}`));
  if(dr===-1) return walls.has(`H-${nr}-${c}`) ||(c>0&&walls.has(`H-${nr}-${c-1}`));
  if(dc===1)  return walls.has(`V-${r}-${c}`)  ||(r>0&&walls.has(`V-${r-1}-${c}`));
  if(dc===-1) return walls.has(`V-${r}-${nc}`) ||(r>0&&walls.has(`V-${r-1}-${nc}`));
  return false;
}
function hasPath(pos,walls,goalRow=0){
  const vis=new Set(),q=[pos];vis.add(ck(pos.r,pos.c));
  while(q.length){const cur=q.shift();if(cur.r===goalRow)return true;
    for(const[dr,dc]of[[-1,0],[1,0],[0,-1],[0,1]]){
      const nr=cur.r+dr,nc=cur.c+dc;
      if(nr>=0&&nr<ROWS&&nc>=0&&nc<COLS){const k=ck(nr,nc);
        if(!vis.has(k)&&!edgeBlocked(cur.r,cur.c,nr,nc,walls)){vis.add(k);q.push({r:nr,c:nc});}}}}
  return false;
}
function bfsPath(pos,walls,goalRow=0){
  const vis=new Map(),q=[pos];vis.set(ck(pos.r,pos.c),null);
  while(q.length){const cur=q.shift();
    if(cur.r===goalRow){const path=[];let k=ck(cur.r,cur.c);
      while(k!==null){const[r,c]=k.split(',').map(Number);path.unshift({r,c});k=vis.get(k);}return path;}
    for(const[dr,dc]of[[-1,0],[1,0],[0,-1],[0,1]]){
      const nr=cur.r+dr,nc=cur.c+dc;
      if(nr>=0&&nr<ROWS&&nc>=0&&nc<COLS){const k=ck(nr,nc);
        if(!vis.has(k)&&!edgeBlocked(cur.r,cur.c,nr,nc,walls)){vis.set(k,ck(cur.r,cur.c));q.push({r:nr,c:nc});}}}}
  return null;
}
function bfsTo(from,target,walls){
  if(from.r===target.r&&from.c===target.c)return[{r:from.r,c:from.c}];
  const vis=new Map(),q=[from];vis.set(ck(from.r,from.c),null);
  while(q.length){const cur=q.shift();
    if(cur.r===target.r&&cur.c===target.c){const path=[];let k=ck(cur.r,cur.c);
      while(k!==null){const[r,c]=k.split(',').map(Number);path.unshift({r,c});k=vis.get(k);}return path;}
    for(const[dr,dc]of[[-1,0],[1,0],[0,-1],[0,1]]){
      const nr=cur.r+dr,nc=cur.c+dc;
      if(nr>=0&&nr<ROWS&&nc>=0&&nc<COLS){const k=ck(nr,nc);
        if(!vis.has(k)&&!edgeBlocked(cur.r,cur.c,nr,nc,walls)){vis.set(k,ck(cur.r,cur.c));q.push({r:nr,c:nc});}}}}
  return null;
}
// ── 2v2 Logik (generisch: Ziel = beliebige Zielreihe, eigene Brettmaße) ──
function hasPathTo(pos,walls,goalRow,R,C){
  const vis=new Set(),q=[pos];vis.add(ck(pos.r,pos.c));
  while(q.length){const cur=q.shift();if(cur.r===goalRow)return true;
    for(const d of[[-1,0],[1,0],[0,-1],[0,1]]){
      const nr=cur.r+d[0],nc=cur.c+d[1];
      if(nr>=0&&nr<R&&nc>=0&&nc<C){const k=ck(nr,nc);
        if(!vis.has(k)&&!edgeBlocked(cur.r,cur.c,nr,nc,walls)){vis.add(k);q.push({r:nr,c:nc});}}}}
  return false;
}
// Anzahl kanten-disjunkter Wege start→Zielreihe via Max-Flow (Edmonds-Karp), gedeckelt bei cap
function disjointPaths2(start,goalRow,walls,R,C,cap){
  const SINK=R*C;const cp=new Map();
  const add=(u,v,c)=>{cp.set(u+'>'+v,(cp.get(u+'>'+v)||0)+c);if(!cp.has(v+'>'+u))cp.set(v+'>'+u,0);};
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){
    const u=r*C+c;
    for(const d of[[1,0],[0,1]]){const nr=r+d[0],nc=c+d[1];
      if(nr<R&&nc<C&&!edgeBlocked(r,c,nr,nc,walls)){const v=nr*C+nc;add(u,v,1);add(v,u,1);}}
    if(r===goalRow)add(u,SINK,1);
  }
  const S=start.r*C+start.c;let flow=0;
  while(flow<cap){
    const prev=new Map();prev.set(S,-1);const q=[S];let found=false;
    while(q.length){const u=q.shift();if(u===SINK){found=true;break;}
      for(let v=0;v<=SINK;v++){const cc=cp.get(u+'>'+v);if(cc&&cc>0&&!prev.has(v)){prev.set(v,u);q.push(v);}}}
    if(!found)break;
    let v=SINK;while(prev.get(v)!==-1){const u=prev.get(v);cp.set(u+'>'+v,cp.get(u+'>'+v)-1);cp.set(v+'>'+u,(cp.get(v+'>'+u)||0)+1);v=u;}
    flow++;
  }
  return flow;
}
// Gueltige Zuege fuer ein Token; occupied = Liste der anderen Token-Positionen (mit Sprung-Regel)
function validMoves2(pos,occupied,walls,R,C){
  const occ=k=>occupied.some(o=>o.r===k.r&&o.c===k.c);
  const moves=[];
  for(const d of[[-1,0],[1,0],[0,-1],[0,1]]){
    const nr=pos.r+d[0],nc=pos.c+d[1];
    if(nr<0||nr>=R||nc<0||nc>=C)continue;
    if(edgeBlocked(pos.r,pos.c,nr,nc,walls))continue;
    if(occ({r:nr,c:nc})){
      const jr=nr+d[0],jc=nc+d[1];
      const straight=jr>=0&&jr<R&&jc>=0&&jc<C&&!edgeBlocked(nr,nc,jr,jc,walls)&&!occ({r:jr,c:jc});
      if(straight){moves.push({r:jr,c:jc});}
      else for(const e of[[d[1],d[0]],[-d[1],-d[0]]]){
        const sr=nr+e[0],sc=nc+e[1];
        if(sr>=0&&sr<R&&sc>=0&&sc<C&&!edgeBlocked(nr,nc,sr,sc,walls)&&!occ({r:sr,c:sc}))moves.push({r:sr,c:sc});
      }
    }else moves.push({r:nr,c:nc});
  }
  return moves;
}
function snapWall2(x,y,orient,R,C,CL){
  if(orient==='H'){const r=Math.min(Math.max(Math.round(y/CL-0.5),0),R-2),c=Math.min(Math.max(Math.floor(x/CL),0),C-2);return 'H-'+r+'-'+c;}
  const c=Math.min(Math.max(Math.round(x/CL-0.5),0),C-2),r=Math.min(Math.max(Math.floor(y/CL),0),R-2);return 'V-'+r+'-'+c;
}
function wallPos2(key,CL){
  const a=key.split('-');const type=a[0],r=+a[1],c=+a[2];
  return type==='H'?{left:c*CL,top:(r+1)*CL-WW2/2,width:2*CL,height:WW2}:{left:(c+1)*CL-WW2/2,top:r*CL,width:WW2,height:2*CL};
}
// Geometrische Mauer-Legalitaet (Ueberlappung)
function wallGeomOk2(key,walls){
  if(walls.has(key))return false;
  const a=key.split('-');const type=a[0],r=+a[1],c=+a[2];
  if(type==='H'){if(walls.has('H-'+r+'-'+(c-1))||walls.has('H-'+r+'-'+(c+1))||walls.has('V-'+r+'-'+c))return false;}
  else{if(walls.has('V-'+(r-1)+'-'+c)||walls.has('V-'+(r+1)+'-'+c)||walls.has('H-'+r+'-'+c))return false;}
  return true;
}
// Token sicher? Keine EINZELNE weitere (geometrisch platzierbare) Barrikade darf ihn vom Ziel abschneiden.
// Begruendung: Barrikade ist 2 Felder breit -> Passage <=2 ist mit einer Mauer schliessbar (unsicher),
// Passage >=3 oder ein getrennter Zweitweg bleibt offen (sicher).
function wallSafeToken2(start,goalRow,walls,R,C){
  if(!hasPathTo(start,walls,goalRow,R,C))return false;
  // Early-out: >=3 kanten-disjunkte Wege -> eine Mauer (2 Kanten) kann nie alle trennen
  if(disjointPaths2(start,goalRow,walls,R,C,3)>=3)return true;
  for(let r=0;r<R-1;r++)for(let c=0;c<C-1;c++){
    let key='H-'+r+'-'+c;
    if(wallGeomOk2(key,walls)){const test=new Set(walls);test.add(key);if(!hasPathTo(start,test,goalRow,R,C))return false;}
    key='V-'+r+'-'+c;
    if(wallGeomOk2(key,walls)){const test=new Set(walls);test.add(key);if(!hasPathTo(start,test,goalRow,R,C))return false;}
  }
  return true;
}
// Schnelle Vorschau (>=1 Weg fuer alle 4) – fuer Drag-Farbe (guenstig, exakte Pruefung beim Drop)
function quickValidWall2(key,walls,tokens,R,C){
  if(!wallGeomOk2(key,walls))return false;
  const next=new Set([...walls,key]);
  for(const id of ORDER_2V2){if(!tokens[id])continue;if(!hasPathTo(tokens[id],next,GOAL_2V2[id],R,C))return false;}
  return true;
}
// Prüft ob ein horizontaler Schnitt bei Reihe r die Breitenregel erfüllt:
// Entweder ≥2 separate Gruppen offener Spalten, oder 1 Gruppe mit ≥3 offenen Spalten.
function passageWidthOk2(walls,r,C){
  const open=[];
  for(let c=0;c<C;c++){
    const blocked=walls.has('H-'+r+'-'+c)||(c>0&&walls.has('H-'+r+'-'+(c-1)));
    if(!blocked)open.push(c);
  }
  if(open.length===0)return false;
  let groups=1,maxGroup=1,cur=1;
  for(let i=1;i<open.length;i++){
    if(open[i]===open[i-1]+1){cur++;if(cur>maxGroup)maxGroup=cur;}
    else{groups++;cur=1;}
  }
  return groups>=2||maxGroup>=3;
}
// Wand-Pruefung wie 1v1: nur Geometrie + jeder Pawn behaelt mindestens EINEN Weg zum Ziel.
function tryWall2(key,walls,tokens,R,C){
  if(!wallGeomOk2(key,walls))return null;
  const next=new Set([...walls,key]);
  for(const id of ORDER_2V2){if(!tokens[id])continue;if(!hasPathTo(tokens[id],next,GOAL_2V2[id],R,C))return null;}
  return next;
}

let _hamCtx=null; // {hammers, seekers:[{r,c}]} – gesetzt vor tryWall, danach null
function buildHamCtx(hammers,hammerHeld,pA,pB){
  if(!hammers||!hammers.length)return null;
  const seekers=[];
  if(hammerHeld&&!hammerHeld.A)seekers.push(pA);
  if(hammerHeld&&!hammerHeld.B)seekers.push(pB);
  return seekers.length?{hammers,seekers}:null;
}
function tryWall(key,walls,pA,pB,goalA=0,goalB=0){
  if(walls.has(key))return null;
  const[type,rs,cs]=key.split('-');const r=+rs,c=+cs;
  if(type==='H'){if(walls.has(`H-${r}-${c-1}`)||walls.has(`H-${r}-${c+1}`))return null;if(walls.has(`V-${r}-${c}`))return null;}
  else{if(walls.has(`V-${r-1}-${c}`)||walls.has(`V-${r+1}-${c}`))return null;if(walls.has(`H-${r}-${c}`))return null;}
  const next=new Set([...walls,key]);
  if(!hasPath(pA,next,goalA)||!hasPath(pB,next,goalB))return null;
  if(_hamCtx&&_hamCtx.seekers.length&&_hamCtx.hammers.length){
    const ok=_hamCtx.hammers.every(h=>_hamCtx.seekers.some(s=>!!bfsTo(s,h,next)));
    if(!ok)return null;
  }
  return next;
}
function snapWall(x,y,orient){
  if(orient==='H'){const r=Math.min(Math.max(Math.round(y/CELL-0.5),0),ROWS-2),c=Math.min(Math.max(Math.floor(x/CELL),0),COLS-2);return `H-${r}-${c}`;}
  const c=Math.min(Math.max(Math.round(x/CELL-0.5),0),COLS-2),r=Math.min(Math.max(Math.floor(y/CELL),0),ROWS-2);return `V-${r}-${c}`;
}
function makeHammers(){
  const r=Math.floor(ROWS/2);
  return[{r,c:1,id:'hm0'},{r,c:Math.floor(COLS/2),id:'hm1'},{r,c:COLS-2,id:'hm2'}];
}
function waveOf(id){return id&&id.indexOf('hw2-')===0?'w2':'w1';}
function freshHammerUsed(){return{A:{w1:false,w2:false},B:{w1:false,w2:false}};}
function isSteelWall(key){
  // Horizontale Mauer direkt vor einem Ziel ist unzerstörbar.
  // Klassisch: nur oben (Reihe 0). Duell: oben (0) UND unten (ROWS-2).
  const parts=key.split('-');
  if(parts[0]!=='H')return false;
  if(parts[1]==='0')return true;
  if(CUR_MAP==='duel'&&parts[1]===String(ROWS-2))return true;
  return false;
}
function makeHammersWave2(pA,pB,walls,goalA=0,goalB=0){
  const pathA=bfsPath(pA,walls,goalA),pathB=bfsPath(pB,walls,goalB);
  if(!pathA||!pathB)return[];
  const dA=pathA.length-1,dB=pathB.length-1;
  const behind=dA>=dB?pA:pB;
  const ahead=dA>=dB?pB:pA;
  const dmBehind=bfsDistMap(behind,walls);
  const dmAhead=bfsDistMap(ahead,walls);
  // Hammer 1: nah am Benachteiligten (2-4 Schritte)
  const cands1=[];
  for(let r=1;r<ROWS-1;r++){for(let c=0;c<COLS;c++){
    if(r===behind.r&&c===behind.c)continue;
    if(r===ahead.r&&c===ahead.c)continue;
    const d=dmBehind.get(ck(r,c));
    if(d===undefined||d<2||d>4)continue;
    cands1.push({r,c,score:d});
  }}
  cands1.sort((a,b)=>a.score-b.score);
  const h1=cands1.length?cands1[0]:null;
  // Hammer 2: mittlere Distanz vom Vorteil-Spieler (5-9 Schritte)
  const cands2=[];
  for(let r=1;r<ROWS-1;r++){for(let c=0;c<COLS;c++){
    if(r===behind.r&&c===behind.c)continue;
    if(r===ahead.r&&c===ahead.c)continue;
    if(h1&&Math.abs(r-h1.r)+Math.abs(c-h1.c)<3)continue;
    const d=dmAhead.get(ck(r,c));
    if(d===undefined||d<5||d>9)continue;
    cands2.push({r,c,score:d});
  }}
  cands2.sort((a,b)=>a.score-b.score);
  const h2=cands2.length?cands2[0]:null;
  const result=[];
  if(h1)result.push({r:h1.r,c:h1.c,id:'hw2-0'});
  if(h2)result.push({r:h2.r,c:h2.c,id:'hw2-1'});
  return result;
}
