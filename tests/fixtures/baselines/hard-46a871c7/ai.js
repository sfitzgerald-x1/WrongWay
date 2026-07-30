// ── ai.js (ausgelagert aus index.html) ────────────────────
// KI-Engine: aiEasy/Normal/Hard/Duel, Chaos/Drop/Hammer-AI.
// Referenziert COLS/ROWS/CELL/WW aus globalem Scope; BFS-Funktionen aus game-logic.js.
function aiHammerBreak(aiPos,walls,goalRow=0){
  const base=bfsPath(aiPos,walls,goalRow);
  if(!base)return null;
  const baseD=base.length-1;
  let best=null,bestGain=0;
  for(const k of walls){
    if(isSteelWall(k))continue;
    const w2=new Set(walls);w2.delete(k);
    const np=bfsPath(aiPos,w2,goalRow);
    if(!np)continue;
    const gain=baseD-(np.length-1);
    if(gain>bestGain){bestGain=gain;best=k;}
  }
  return bestGain>=2?best:null;
}
function nearestHammer(aiPos,walls,hammers){
  if(!hammers||!hammers.length)return null;
  let best=null,bestLen=Infinity;
  for(const h of hammers){
    const p=bfsTo(aiPos,h,walls);
    if(p&&(p.length-1)<bestLen){bestLen=p.length-1;best={h,path:p};}
  }
  return best;
}
function wallPos(key){
  const[type,rs,cs]=key.split('-');const r=+rs,c=+cs;
  return type==='H'?{left:c*CELL,top:(r+1)*CELL-WW/2,width:2*CELL,height:WW}:{left:(c+1)*CELL-WW/2,top:r*CELL,width:WW,height:2*CELL};
}

// ── Chaos: faire Drop-Position ─────────────────────────────
function bfsDistMap(from,walls){
  const dist=new Map();
  const q=[from];
  dist.set(ck(from.r,from.c),0);
  while(q.length){
    const cur=q.shift();
    const d=dist.get(ck(cur.r,cur.c));
    for(const[dr,dc]of[[-1,0],[1,0],[0,-1],[0,1]]){
      const nr=cur.r+dr,nc=cur.c+dc;
      if(nr<0||nr>=ROWS||nc<0||nc>=COLS)continue;
      const k=ck(nr,nc);
      if(dist.has(k))continue;
      if(edgeBlocked(cur.r,cur.c,nr,nc,walls))continue;
      dist.set(k,d+1);
      q.push({r:nr,c:nc});
    }
  }
  return dist;
}
function pickChaosPos(pA,pB,walls,goalA=0,goalB=0){
  const pa=bfsPath(pA,walls,goalA),pb=bfsPath(pB,walls,goalB);
  if(!pa||!pb)return null;
  const lenA=pa.length-1,lenB=pb.length-1;
  // Faire Zone: vertikal zwischen den beiden Spielern (richtungsneutral),
  // mit 1 Reihe Abstand zu beiden Ziellinien, nie hinter/auf einem Spieler.
  const loR=Math.min(pA.r,pB.r),hiR=Math.max(pA.r,pB.r);
  const minR=Math.max(1,loR);
  const maxR=Math.min(ROWS-2,hiR);
  if(maxR<minR)return null;
  const inZone=p=>{
    if(p.r<minR||p.r>maxR)return false;
    if(p.r===pA.r&&p.c===pA.c)return false;
    if(p.r===pB.r&&p.c===pB.c)return false;
    return true;
  };

  // Ungleichgewicht > 1: Item DIREKT auf BFS-Pfad des Schwächeren platzieren
  // → garantiert erreichbar, kein Detour vom Ziel, nicht hinter Mauern
  if(Math.abs(lenA-lenB)>1){
    const weakPath=lenA>lenB?pa:pb;
    // Bevorzugt 2–4 Schritte voraus auf seinem Pfad
    for(let k=2;k<=Math.min(4,weakPath.length-1);k++){
      if(inZone(weakPath[k]))return{r:weakPath[k].r,c:weakPath[k].c};
    }
    // Fallback: erster Pfad-Schritt der in Zone passt
    for(let k=1;k<weakPath.length;k++){
      if(inZone(weakPath[k]))return{r:weakPath[k].r,c:weakPath[k].c};
    }
  }

  // Gleichstand: Position die für beide BFS-erreichbar und ähnlich nah ist
  const dmA=bfsDistMap(pA,walls);
  const dmB=bfsDistMap(pB,walls);
  let best=null,bestScore=Infinity;
  for(let r=minR;r<=maxR;r++){
    for(let c=0;c<COLS;c++){
      if(!inZone({r,c}))continue;
      const dA=dmA.get(ck(r,c)),dB=dmB.get(ck(r,c));
      if(dA===undefined||dB===undefined)continue; // unerreichbar (hinter Mauer)
      // Kleine Summe + kleine Differenz = fair & nicht zu weit
      const score=dA+dB+Math.abs(dA-dB)*3;
      if(score<bestScore){bestScore=score;best={r,c};}
    }
  }
  return best;
}

// ── Random-Drop: Intervall je Häufigkeit ───────────────────
function dropInterval(mode){
  if(mode==='rare')return 30+Math.floor(Math.random()*11);     // 30–40
  if(mode==='often')return 20+Math.floor(Math.random()*6);     // 20–25
  if(mode==='veryoften')return 10+Math.floor(Math.random()*6); // 10–15
  return 99999;
}
// ── Random-Drop: faire Wand wählen ─────────────────────────
// Behindert eher den Führenden, niemals den Hinterherlaufenden stärker;
// tryWall garantiert, dass beide immer einen Weg zum Ziel behalten.
function pickFairDropWall(pA,pB,walls,goalA=0,goalB=0){
  const pa=bfsPath(pA,walls,goalA),pb=bfsPath(pB,walls,goalB);
  if(!pa||!pb)return null;
  const lenA=pa.length-1,lenB=pb.length-1;
  const leaderIsA=lenA<=lenB;                 // kleinere Distanz = näher am Ziel = Vorteil
  const leaderPos=leaderIsA?pA:pB;
  const trailerPos=leaderIsA?pB:pA;
  const leaderGoal=leaderIsA?goalA:goalB;
  const trailerGoal=leaderIsA?goalB:goalA;
  const lenLeader=leaderIsA?lenA:lenB;
  const lenTrailer=leaderIsA?lenB:lenA;
  const cands=wallCandidates(leaderPos,walls,6,leaderGoal);
  const scored=[];
  for(const key of cands){
    const next=tryWall(key,walls,pA,pB,goalA,goalB); // null wenn ungültig / sperrt jemanden ein
    if(!next)continue;
    const npL=bfsPath(leaderPos,next,leaderGoal),npT=bfsPath(trailerPos,next,trailerGoal);
    if(!npL||!npT)continue;
    const dLeader=(npL.length-1)-lenLeader;
    const dTrailer=(npT.length-1)-lenTrailer;
    if(dTrailer>dLeader)continue;             // Fairness: Schwächeren nie stärker behindern
    const score=dLeader-2.5*Math.max(0,dTrailer)+Math.random()*1.5;
    scored.push({key,score});
  }
  if(!scored.length)return null;
  scored.sort((a,b)=>b.score-a.score);
  const top=scored.slice(0,Math.min(5,scored.length)); // Top-5 → Zufall = unvorhersehbar
  return top[Math.floor(Math.random()*top.length)].key;
}
function getMovesFrom(pos,opp,walls){
  const moves=[];
  for(const[dr,dc]of[[-1,0],[1,0],[0,-1],[0,1]]){
    const nr=pos.r+dr,nc=pos.c+dc;
    if(nr<0||nr>=ROWS||nc<0||nc>=COLS)continue;
    if(edgeBlocked(pos.r,pos.c,nr,nc,walls))continue;
    if(nr===opp.r&&nc===opp.c){
      for(const[d2r,d2c]of[[-1,0],[1,0],[0,-1],[0,1]]){
        const jr=nr+d2r,jc=nc+d2c;
        if(jr===pos.r&&jc===pos.c)continue;
        if(jr<0||jr>=ROWS||jc<0||jc>=COLS)continue;
        if(edgeBlocked(nr,nc,jr,jc,walls))continue;
        moves.push({r:jr,c:jc});
      }
    }else moves.push({r:nr,c:nc});
  }
  return moves;
}

function wallCandidates(targetPos,walls,depth,goalRow=0){
  const path=bfsPath(targetPos,walls,goalRow);
  const result=new Set();
  if(!path)return[];
  const limit=Math.min(depth,path.length);
  for(let i=0;i<limit;i++){
    const node=path[i];
    for(let dr=-1;dr<=0;dr++)for(let dc=-1;dc<=0;dc++){
      const r=node.r+dr,c=node.c+dc;
      if(r>=0&&r<ROWS-1&&c>=0&&c<COLS-1){
        result.add(`H-${r}-${c}`);
        result.add(`V-${r}-${c}`);
      }
    }
  }
  return[...result];
}

function shouldPickupItem(aiPos,humanPos,walls,aiPath,chaosItem,aiBarr,aiGoal=0){
  if(!chaosItem||!aiPath)return null;
  // Item direkt erreichbar?
  const moves=getMovesFrom(aiPos,humanPos,walls);
  const direct=moves.find(m=>m.r===chaosItem.r&&m.c===chaosItem.c);
  if(direct)return direct; // sofort einsacken
  // Detour-Berechnung: AI → Item → Ziel vs AI → Ziel
  const toItem=bfsTo(aiPos,chaosItem,walls);
  if(!toItem||toItem.length<2)return null;
  const fromItem=bfsPath(chaosItem,walls,aiGoal);
  if(!fromItem)return null;
  const itemDist=toItem.length-1;
  const detour=(itemDist+(fromItem.length-1))-(aiPath.length-1);
  // 2 Barrikaden ≈ 4-6 Zugäquivalente; je weniger Barrikaden, desto attraktiver
  const maxDetour=aiBarr<=2?6:aiBarr<=4?4:aiBarr<=7?2:1;
  if(detour>maxDetour)return null;
  const next=toItem[1];
  if(next.r===humanPos.r&&next.c===humanPos.c)return null;
  return next;
}

function aiMove(aiPos,humanPos,walls,aiBarr,humBarr,difficulty,recentAi,chaosItem,aiGoal=0,humGoal=0){
  difficulty=difficulty||'normal';
  if(difficulty==='hard')return aiHard(aiPos,humanPos,walls,aiBarr,humBarr,recentAi,chaosItem,aiGoal,humGoal);
  if(difficulty==='easy')return aiEasy(aiPos,humanPos,walls,aiBarr,chaosItem,aiGoal,humGoal);
  return aiNormal(aiPos,humanPos,walls,aiBarr,chaosItem,aiGoal,humGoal);
}

function moveTowardGoal(aiPos,humanPos,walls,aiPath){
  if(!aiPath||aiPath.length<2)return null;
  const ns=aiPath[1];
  if(ns.r!==humanPos.r||ns.c!==humanPos.c)return ns;
  if(edgeBlocked(aiPos.r,aiPos.c,ns.r,ns.c,walls))return null;
  const dr=ns.r-aiPos.r,dc=ns.c-aiPos.c;
  for(const[d2r,d2c]of[[dr,dc],[-dc,dr],[dc,-dr],[-dr,-dc]]){
    const jr=ns.r+d2r,jc=ns.c+d2c;
    if(jr===aiPos.r&&jc===aiPos.c)continue;
    if(jr<0||jr>=ROWS||jc<0||jc>=COLS)continue;
    if(edgeBlocked(ns.r,ns.c,jr,jc,walls))continue;
    return{r:jr,c:jc};
  }
  return null;
}

function aiEasy(aiPos,humanPos,walls,barrLeft,chaosItem,aiGoal=0,humGoal=0){
  // Schrittrichtung Richtung eigenes Ziel (oben=-1, unten=+1)
  const fwd=aiGoal>aiPos.r?1:-1;
  const aiPath=bfsPath(aiPos,walls,aiGoal);
  const humPath=bfsPath(humanPos,walls,humGoal);
  if(!aiPath)return null;
  const aiD=aiPath.length-1,humD=humPath?humPath.length-1:999;
  if(aiD===1)return{type:'move',pos:aiPath[1]};

  // Easy: nimmt das Item nur wenn quasi gratis (Detour-Toleranz in shouldPickupItem niedrig)
  if(chaosItem){
    const pick=shouldPickupItem(aiPos,humanPos,walls,aiPath,chaosItem,barrLeft,aiGoal);
    if(pick)return{type:'move',pos:pick};
  }

  // Easy: Setzt jetzt deutlich häufiger Mauern, aber wählt suboptimal
  // Mauer-Bedingungen gelockert: Gegner führt oder ist im vorderen Drittel
  const shouldConsiderWall=barrLeft>0&&(humD<=aiD+3||humD<=6);
  // Zufällige Chance, manchmal Mauer auch ohne starken Grund
  const proactive=barrLeft>0&&humD<=aiD+5&&Math.random()<0.45;
  if(shouldConsiderWall||proactive){
    const cands=wallCandidates(humanPos,walls,5,humGoal);
    // Zufällige Reihenfolge → suboptimale Wahl
    for(let i=cands.length-1;i>0;i--){
      const j=Math.floor(Math.random()*(i+1));
      [cands[i],cands[j]]=[cands[j],cands[i]];
    }
    for(const k of cands){
      const next=tryWall(k,walls,aiPos,humanPos,aiGoal,humGoal);
      if(!next)continue;
      const nh=bfsPath(humanPos,next,humGoal),na=bfsPath(aiPos,next,aiGoal);
      if(!nh||!na)continue;
      // Gelockert: Gegner muss um ≥1 verlangsamt werden, AI darf max ≤2 leiden
      if((nh.length-1)-humD>=1&&(na.length-1)-aiD<=2){
        return{type:'barricade',walls:next};
      }
    }
  }
  const m=moveTowardGoal(aiPos,humanPos,walls,aiPath);
  if(m)return{type:'move',pos:m};
  const fallback=getMovesFrom(aiPos,humanPos,walls);
  return fallback.length?{type:'move',pos:fallback[0]}:null;
}

function aiNormal(aiPos,humanPos,walls,barrLeft,chaosItem,aiGoal=0,humGoal=0){
  const aiPath=bfsPath(aiPos,walls,aiGoal),humPath=bfsPath(humanPos,walls,humGoal);
  if(!aiPath)return null;
  const aiD=aiPath.length-1,humD=humPath?humPath.length-1:999;
  if(aiD===1)return{type:'move',pos:aiPath[1]};
  // Item einsammeln wenn lohnenswert
  if(chaosItem){
    const pick=shouldPickupItem(aiPos,humanPos,walls,aiPath,chaosItem,barrLeft,aiGoal);
    if(pick)return{type:'move',pos:pick};
  }
  let bestWall=null,bestRel=humD-aiD;
  if(barrLeft>0){
    for(let r=0;r<ROWS-1;r++)for(let c=0;c<COLS-1;c++)for(const t of['H','V']){
      const next=tryWall(`${t}-${r}-${c}`,walls,aiPos,humanPos,aiGoal,humGoal);
      if(!next)continue;
      const nh=bfsPath(humanPos,next,humGoal),na=bfsPath(aiPos,next,aiGoal);
      if(!nh||!na)continue;
      const rel=(nh.length-1)-(na.length-1);
      if(rel>bestRel){bestRel=rel;bestWall=next;}
    }
  }
  const bestMove=moveTowardGoal(aiPos,humanPos,walls,aiPath);
  const minGain=(humD-aiD)>3?3:1;
  if(bestWall&&bestRel-(humD-aiD)>=minGain)return{type:'barricade',walls:bestWall};
  if(bestMove)return{type:'move',pos:bestMove};
  if(bestWall)return{type:'barricade',walls:bestWall};
  return null;
}

// ── Aggressive Duell-KI (nur Duell-Modus, 7x7 & 9x9) ──────────────
// Distanz = kürzester Pfad (BFS = A*-Ergebnis auf ungewichtetem Grid).
// Bewertet jede sinnvolle Mauer und wählt strategisch zwischen Blocken und Vordrang.
function duelDist(pos,walls,goal){const p=bfsPath(pos,walls,goal);return p?p.length-1:Infinity;}
function duelBestWall(aiPos,humPos,walls,aiGoal,humGoal){
  // Kandidaten: Mauern entlang des Gegnerpfads (verlangsamen ihn maximal)
  const cand=wallCandidates(humPos,walls,5,humGoal);
  const baseHum=duelDist(humPos,walls,humGoal);
  const baseAi=duelDist(aiPos,walls,aiGoal);
  let best=null,bestScore=-Infinity;
  for(const key of cand){
    const nw=tryWall(key,walls,aiPos,humPos,aiGoal,humGoal); // null = ungültig / sperrt jemanden ein
    if(!nw)continue;
    const newHum=duelDist(humPos,nw,humGoal);
    const newAi=duelDist(aiPos,nw,aiGoal);
    if(newHum===Infinity||newAi===Infinity)continue;
    const humDelay=newHum-baseHum;   // wie sehr Gegner verlangsamt wird (gut)
    const aiCost=newAi-baseAi;       // wie sehr man sich selbst behindert (schlecht)
    if(humDelay<=0)continue;         // nur Mauern die wirklich blocken
    // Score: Gegner-Verzögerung stark gewichten, Eigenschaden doppelt bestrafen
    const score=humDelay*3-aiCost*2;
    if(score>bestScore){bestScore=score;best={key,walls:nw,humDelay,aiCost};}
  }
  return best;
}
function aiDuel(aiPos,humanPos,walls,aiBarr,humBarr,recentAi,chaosItem,aiGoal,humGoal){
  const recent=recentAi||[];
  const initMoves=getMovesFrom(aiPos,humanPos,walls);
  // 1) Sofortiger Sieg
  for(const m of initMoves)if(m.r===aiGoal)return{type:'move',pos:m};
  // Item direkt erreichbar mitnehmen
  if(chaosItem){const d=initMoves.find(m=>m.r===chaosItem.r&&m.c===chaosItem.c);if(d)return{type:'move',pos:d};}

  const aiPath=bfsPath(aiPos,walls,aiGoal);
  const humPath=bfsPath(humanPos,walls,humGoal);
  if(!aiPath)return{type:'move',pos:initMoves[0]||aiPos};
  const aiD=aiPath.length-1;
  const humD=humPath?humPath.length-1:999;
  const advance=()=>{
    const m=moveTowardGoal(aiPos,humanPos,walls,aiPath);
    if(m){
      const wasRecent=recent.slice(0,4).some(p=>p.r===m.r&&p.c===m.c);
      if(!wasRecent)return{type:'move',pos:m};
      const alt=initMoves.filter(mv=>!recent.slice(0,4).some(p=>p.r===mv.r&&p.c===mv.c));
      if(alt.length>0)return{type:'move',pos:alt[0]};
    }
    return m?{type:'move',pos:m}:{type:'move',pos:(initMoves[0]||aiPath[1])};
  };

  // 2) Gegner führt (humD < aiD): aggressiv blocken (70%)
  if(humD<aiD&&aiBarr>0){
    const block=duelBestWall(aiPos,humanPos,walls,aiGoal,humGoal);
    if(block){
      // 70% blocken wenn die Mauer wirklich etwas bringt; sonst vorrücken
      const wantBlock=Math.random()<0.7&&block.humDelay>=1;
      if(wantBlock)return{type:'barricade',key:block.key,walls:block.walls};
    }
    return advance();
  }

  // 3) KI führt oder gleichauf: clever vordrängen, aber Gegner einmauern wenn extrem effektiv
  if(aiBarr>0){
    const block=duelBestWall(aiPos,humanPos,walls,aiGoal,humGoal);
    // Nur blocken wenn es den Gegner deutlich (>=2) verzögert und kaum eigenen Schaden macht
    if(block&&block.humDelay>=2&&block.aiCost<=1&&Math.random()<0.45){
      return{type:'barricade',key:block.key,walls:block.walls};
    }
    // Bei sehr knappem Vorsprung des Gegners zumindest leicht blocken
    if(block&&humD<=aiD+1&&block.humDelay>=1&&block.aiCost<=0&&Math.random()<0.35){
      return{type:'barricade',key:block.key,walls:block.walls};
    }
  }
  return advance();
}

// ── Duell-KI v2: klare Abstufung (Normal solide, Hard nahezu unschlagbar) ──
// Misst wie stark ein einzelner Gegnerzug den eigenen Weg verzoegern koennte (klein = robuster).
function worstWallDelay(target,walls,goal,pA,pB,goalA,goalB){
  const base=duelDist(target,walls,goal);
  if(base===Infinity)return 99;
  const cands=wallCandidates(target,walls,4,goal);
  let worst=0;
  for(const k of cands){
    const nw=tryWall(k,walls,pA,pB,goalA,goalB);
    if(!nw)continue;
    const nd=duelDist(target,nw,goal);
    if(nd===Infinity)continue;
    const delay=nd-base;
    if(delay>worst)worst=delay;
  }
  return worst;
}
// Anzahl kanten-disjunkter Wege zum Ziel (mehr = schwerer mit einer Mauer zu kappen).
function aiDisjoint(start,goal,walls,cap){
  try{return disjointPaths2(start,goal,walls,ROWS,COLS,cap);}catch(_){return 1;}
}

// Duell-Normal: 1-Zug-Vorausschau, blockt optimal gewaehlt & rueckt klug vor.
// Deutlich ueber Easy (optimale Mauerwahl statt zufaellig), klar unter Hard (kein Minimax).
function aiDuelNormal(aiPos,humanPos,walls,aiBarr,humBarr,recentAi,chaosItem,aiGoal,humGoal){
  const initMoves=getMovesFrom(aiPos,humanPos,walls);
  for(const m of initMoves)if(m.r===aiGoal)return{type:'move',pos:m};
  if(chaosItem){const d=initMoves.find(m=>m.r===chaosItem.r&&m.c===chaosItem.c);if(d)return{type:'move',pos:d};}
  const aiPath=bfsPath(aiPos,walls,aiGoal),humPath=bfsPath(humanPos,walls,humGoal);
  if(!aiPath)return{type:'move',pos:initMoves[0]||aiPos};
  const aiD=aiPath.length-1,humD=humPath?humPath.length-1:999;
  const advance=()=>{const m=moveTowardGoal(aiPos,humanPos,walls,aiPath);return m?{type:'move',pos:m}:{type:'move',pos:(initMoves[0]||aiPath[1])};};
  if(chaosItem){const pick=shouldPickupItem(aiPos,humanPos,walls,aiPath,chaosItem,aiBarr,aiGoal);if(pick)return{type:'move',pos:pick};}
  const block=aiBarr>0?duelBestWall(aiPos,humanPos,walls,aiGoal,humGoal):null;
  // Gegner fuehrt oder gleichauf -> aggressiv blocken
  if(block&&humD<=aiD&&block.humDelay>=1&&block.aiCost<=1){
    const p=humD<aiD?0.85:0.55;
    if(Math.random()<p)return{type:'barricade',key:block.key,walls:block.walls};
  }
  // KI fuehrt -> nur sehr effiziente Mauern setzen
  if(block&&block.humDelay>=2&&block.aiCost<=0&&Math.random()<0.4){
    return{type:'barricade',key:block.key,walls:block.walls};
  }
  return advance();
}

// Duell-Hard: Minimax (Alpha-Beta, Tiefe 3) + Pfad-Robustheit.
// Sichert den eigenen Weg ab (schwer blockierbar) und blockt den Gegner vorausschauend.
function aiDuelHard(aiPos,humanPos,walls,aiBarr,humBarr,recentAi,chaosItem,aiGoal,humGoal){
  const initMoves=getMovesFrom(aiPos,humanPos,walls);
  for(const m of initMoves)if(m.r===aiGoal)return{type:'move',pos:m};
  if(chaosItem){const direct=initMoves.find(m=>m.r===chaosItem.r&&m.c===chaosItem.c);if(direct)return{type:'move',pos:direct};}
  const recent=recentAi||[];
  const aiFwdDown=aiGoal>aiPos.r;
  const aiPathInit=bfsPath(aiPos,walls,aiGoal);
  const humPathInit=bfsPath(humanPos,walls,humGoal);

  // Endspiel-Sprint: klar in Front & sehr kurzer Pfad -> direkt durchlaufen.
  if(aiPathInit&&aiPathInit.length>1){
    const aiDI=aiPathInit.length-1;
    const humDI=humPathInit?humPathInit.length-1:999;
    if((aiBarr<=0&&!chaosItem)||(aiDI<=2&&humDI>aiDI)){
      const m=moveTowardGoal(aiPos,humanPos,walls,aiPathInit);
      if(m)return{type:'move',pos:m};
    }
  }

  const t0=Date.now();
  const TIME=700;

  const evalState=(ap,hp,w,ab,hb)=>{
    const apath=bfsPath(ap,w,aiGoal),hpath=bfsPath(hp,w,humGoal);
    if(!apath)return-99999;
    if(!hpath)return 99999;
    if(ap.r===aiGoal)return 50000;
    if(hp.r===humGoal)return-50000;
    const aiD=apath.length-1,humD=hpath.length-1;
    return(humD-aiD)*28 - aiD*4 + humD*1.0 + (ab-hb)*5;
  };

  const genActions=(ap,hp,w,ab,hb,forAi,maxWalls)=>{
    const me=forAi?ap:hp;
    const opp=forAi?hp:ap;
    const myBarr=forAi?ab:hb;
    const meGoal=forAi?aiGoal:humGoal;
    const oppGoal=forAi?humGoal:aiGoal;
    const acts=[];
    const myPath=bfsPath(me,w,meGoal);
    const moves=getMovesFrom(me,opp,w);
    for(const m of moves){
      let pri=0;
      if(myPath&&myPath.length>1&&myPath[1].r===m.r&&myPath[1].c===m.c)pri=30;
      if(meGoal>me.r?m.r>me.r:m.r<me.r)pri+=8;
      acts.push({type:'move',pos:m,_p:pri});
    }
    if(myBarr>0){
      const cands=wallCandidates(opp,w,6,oppGoal);
      const oldOppPath=bfsPath(opp,w,oppGoal);
      const oldOppD=oldOppPath?oldOppPath.length-1:0;
      const myOldD=myPath?myPath.length-1:0;
      const scored=[];
      for(const k of cands){
        const next=tryWall(k,w,ap,hp,aiGoal,humGoal);
        if(!next)continue;
        const np=bfsPath(opp,next,oppGoal);
        if(!np)continue;
        const myNew=bfsPath(me,next,meGoal);
        if(!myNew)continue;
        const gain=((np.length-1)-oldOppD)-((myNew.length-1)-myOldD);
        if(gain<=0)continue;
        scored.push({next,gain});
      }
      scored.sort((a,b)=>b.gain-a.gain);
      const lim=Math.min(maxWalls,scored.length);
      for(let i=0;i<lim;i++)acts.push({type:'barricade',walls:scored[i].next,_p:scored[i].gain*8});
    }
    acts.sort((a,b)=>b._p-a._p);
    return acts;
  };

  const search=(ap,hp,w,ab,hb,depth,maximizing,alpha,beta)=>{
    if(ap.r===aiGoal)return 50000-depth;
    if(hp.r===humGoal)return-50000+depth;
    if(depth===0||Date.now()-t0>TIME)return evalState(ap,hp,w,ab,hb);
    const maxW=depth>=2?4:3;
    const acts=genActions(ap,hp,w,ab,hb,maximizing,maxW);
    if(acts.length===0)return evalState(ap,hp,w,ab,hb);
    if(maximizing){
      let best=-Infinity;
      for(const a of acts){
        const s=a.type==='move'
          ?search(a.pos,hp,w,ab,hb,depth-1,false,alpha,beta)
          :search(ap,hp,a.walls,ab-1,hb,depth-1,false,alpha,beta);
        if(s>best)best=s;
        if(best>alpha)alpha=best;
        if(beta<=alpha)break;
      }
      return best;
    }
    let best=Infinity;
    for(const a of acts){
      const s=a.type==='move'
        ?search(ap,a.pos,w,ab,hb,depth-1,true,alpha,beta)
        :search(ap,hp,a.walls,ab,hb-1,depth-1,true,alpha,beta);
      if(s<best)best=s;
      if(best<beta)beta=best;
      if(beta<=alpha)break;
    }
    return best;
  };

  let bestScore=-Infinity,bestAction=null;
  const myActs=genActions(aiPos,humanPos,walls,aiBarr,humBarr,true,8);
  for(const a of myActs){
    let s=a.type==='move'
      ?search(a.pos,humanPos,walls,aiBarr,humBarr,3,false,-Infinity,Infinity)
      :search(aiPos,humanPos,a.walls,aiBarr-1,humBarr,3,false,-Infinity,Infinity);
    // Resultierender Zustand fuer Robustheits-Bewertung (eigenen Weg schuetzen)
    const rPos=a.type==='move'?a.pos:aiPos;
    const rW=a.type==='move'?walls:a.walls;
    const worst=worstWallDelay(rPos,rW,aiGoal,rPos,humanPos,aiGoal,humGoal);
    s-=worst*9;                                  // schwer blockierbarer Weg = besser
    s+=aiDisjoint(rPos,aiGoal,rW,3)*6;           // mehr Ausweichrouten = besser
    if(a.type==='move'){
      if(aiPathInit&&aiPathInit.length>1&&aiPathInit[1].r===a.pos.r&&aiPathInit[1].c===a.pos.c)s+=20;
      if(aiFwdDown?a.pos.r>aiPos.r:a.pos.r<aiPos.r)s+=6;
      else if(aiFwdDown?a.pos.r<aiPos.r:a.pos.r>aiPos.r)s-=4;
      if(chaosItem){
        const oldD=Math.abs(aiPos.r-chaosItem.r)+Math.abs(aiPos.c-chaosItem.c);
        const newD=Math.abs(a.pos.r-chaosItem.r)+Math.abs(a.pos.c-chaosItem.c);
        if(newD<oldD&&newD<=4)s+=(5-newD)*4;
        if(newD===0)s+=30;
      }
      if(recent.length>=2){
        for(let i=1;i<Math.min(recent.length,5);i++){
          if(recent[i].r===a.pos.r&&recent[i].c===a.pos.c){s-=(6-i)*12;break;}
        }
      }
    }
    if(s>bestScore){bestScore=s;bestAction=a;}
  }
  if(!bestAction){
    if(aiPathInit&&aiPathInit.length>1)return{type:'move',pos:aiPathInit[1]};
    const fm=getMovesFrom(aiPos,humanPos,walls);
    if(fm.length)return{type:'move',pos:fm[0]};
    return null;
  }
  if(bestAction.type==='move')return{type:'move',pos:bestAction.pos};
  return{type:'barricade',walls:bestAction.walls};
}

function aiHard(aiPos,humanPos,walls,aiBarr,humBarr,recentAi,chaosItem,aiGoal=0,humGoal=0){
  const initMoves=getMovesFrom(aiPos,humanPos,walls);
  // Sofortiger Sieg hat absolute Priorität (eigene Zielreihe erreichen)
  for(const m of initMoves)if(m.r===aiGoal)return{type:'move',pos:m};
  // Item-Pickup wenn direkt erreichbar (sehr lohnenswert) — vor Minimax
  if(chaosItem){
    const direct=initMoves.find(m=>m.r===chaosItem.r&&m.c===chaosItem.c);
    if(direct)return{type:'move',pos:direct};
  }
  const recent=recentAi||[];
  // Vorwärts = Richtung eigenes Ziel
  const aiFwdDown=aiGoal>aiPos.r;

  // Endspiel-Sprint: bei kurzem Pfad und klarem Vorsprung direkt zum Ziel laufen,
  // statt unnötig Mauern zu suchen oder im Minimax-Rauschen zu trödeln.
  const aiPathInit=bfsPath(aiPos,walls,aiGoal);
  const humPathInit=bfsPath(humanPos,walls,humGoal);
  if(aiPathInit&&aiPathInit.length>1){
    const aiDI=aiPathInit.length-1;
    const humDI=humPathInit?humPathInit.length-1:999;
    // Reine Zielverfolgung (deterministisch, kein Kreiseln) wenn:
    // - keine Mauern mehr zum Setzen (und kein lohnendes Item), ODER
    // - kurzer Pfad mit Vorsprung, ODER
    // - klar in Führung
    if((aiBarr<=0&&!chaosItem)||(aiDI<=3&&humDI>aiDI+1)||humDI>aiDI+2){
      const m=moveTowardGoal(aiPos,humanPos,walls,aiPathInit);
      if(m)return{type:'move',pos:m};
    }
  }

  // Statische Evaluation: positiver Score = gut für AI
  const evalState=(ap,hp,w,ab,hb)=>{
    const apath=bfsPath(ap,w,aiGoal),hpath=bfsPath(hp,w,humGoal);
    if(!apath)return-99999;
    if(!hpath)return 99999;
    if(ap.r===aiGoal)return 50000;
    if(hp.r===humGoal)return-50000;
    const aiD=apath.length-1,humD=hpath.length-1;
    // Hauptfaktor: relative Distanz
    // Eigener Progress STARK belohnt (verhindert Trödeln/Hin-und-Her)
    // Gegner-Distanz leichter Bonus
    // Mauern-Reserve sekundär
    return(humD-aiD)*25 - aiD*3.5 + humD*0.8 + (ab-hb)*3;
  };

  // Aktionen für aktuellen Spieler generieren, sortiert für Alpha-Beta-Effizienz
  const genActions=(ap,hp,w,ab,hb,forAi,maxWalls)=>{
    const me=forAi?ap:hp;
    const opp=forAi?hp:ap;
    const myBarr=forAi?ab:hb;
    const meGoal=forAi?aiGoal:humGoal;
    const oppGoal=forAi?humGoal:aiGoal;
    const acts=[];
    const myPath=bfsPath(me,w,meGoal);
    const moves=getMovesFrom(me,opp,w);
    for(const m of moves){
      let pri=0;
      if(myPath&&myPath.length>1&&myPath[1].r===m.r&&myPath[1].c===m.c)pri=30;
      // Bonus für Schritt Richtung eigenes Ziel
      if(meGoal>me.r?m.r>me.r:m.r<me.r)pri+=8;
      acts.push({type:'move',pos:m,_p:pri});
    }
    if(myBarr>0){
      const cands=wallCandidates(opp,w,6,oppGoal);
      const oldOppPath=bfsPath(opp,w,oppGoal);
      const oldOppD=oldOppPath?oldOppPath.length-1:0;
      const scored=[];
      for(const k of cands){
        const next=tryWall(k,w,ap,hp,aiGoal,humGoal);
        if(!next)continue;
        const np=bfsPath(opp,next,oppGoal);
        if(!np)continue;
        const myNew=bfsPath(me,next,meGoal);
        if(!myNew)continue;
        // Score: wieviel verlangsamt es Gegner abzüglich Eigenkosten
        const gain=((np.length-1)-oldOppD)-((myNew.length-1)-(myPath?myPath.length-1:0));
        if(gain<=0)continue; // Nur Mauern die wirklich helfen
        scored.push({next,gain});
      }
      scored.sort((a,b)=>b.gain-a.gain);
      const lim=Math.min(maxWalls,scored.length);
      for(let i=0;i<lim;i++){
        acts.push({type:'barricade',walls:scored[i].next,_p:scored[i].gain*8});
      }
    }
    acts.sort((a,b)=>b._p-a._p);
    return acts;
  };

  // Minimax mit Alpha-Beta
  const search=(ap,hp,w,ab,hb,depth,maximizing,alpha,beta)=>{
    if(ap.r===aiGoal)return 50000-depth;
    if(hp.r===humGoal)return-50000+depth;
    if(depth===0)return evalState(ap,hp,w,ab,hb);
    const maxW=depth>=2?6:4;
    const acts=genActions(ap,hp,w,ab,hb,maximizing,maxW);
    if(acts.length===0)return evalState(ap,hp,w,ab,hb);
    if(maximizing){
      let best=-Infinity;
      for(const a of acts){
        const s=a.type==='move'
          ?search(a.pos,hp,w,ab,hb,depth-1,false,alpha,beta)
          :search(ap,hp,a.walls,ab-1,hb,depth-1,false,alpha,beta);
        if(s>best)best=s;
        if(best>alpha)alpha=best;
        if(beta<=alpha)break;
      }
      return best;
    }
    let best=Infinity;
    for(const a of acts){
      const s=a.type==='move'
        ?search(ap,a.pos,w,ab,hb,depth-1,true,alpha,beta)
        :search(ap,hp,a.walls,ab,hb-1,depth-1,true,alpha,beta);
      if(s<best)best=s;
      if(best<beta)beta=best;
      if(beta<=alpha)break;
    }
    return best;
  };

  // Wurzel: 3 Halbzüge tief suchen (AI → Opp → AI → eval)
  let bestScore=-Infinity,bestAction=null;
  const myActs=genActions(aiPos,humanPos,walls,aiBarr,humBarr,true,10);
  for(const a of myActs){
    let s=a.type==='move'
      ?search(a.pos,humanPos,walls,aiBarr,humBarr,2,false,-Infinity,Infinity)
      :search(aiPos,humanPos,a.walls,aiBarr-1,humBarr,2,false,-Infinity,Infinity);
    if(a.type==='move'){
      // Starker Bias auf den BFS-Hauptpfad: verhindert Wechsel zwischen gleich­langen
      // Pfaden, wodurch die AI sonst zickzack laufen würde.
      if(aiPathInit&&aiPathInit.length>1&&aiPathInit[1].r===a.pos.r&&aiPathInit[1].c===a.pos.c){
        s+=20;
      }
      // Zusätzlicher Bonus für tatsächliche Annäherung Richtung eigenes Ziel
      if(aiFwdDown?a.pos.r>aiPos.r:a.pos.r<aiPos.r)s+=6;
      else if(aiFwdDown?a.pos.r<aiPos.r:a.pos.r>aiPos.r)s-=4;
      // Bonus für Annäherung ans Chaos-Item (proportional zum Wert von +2 Barrikaden)
      if(chaosItem){
        const oldD=Math.abs(aiPos.r-chaosItem.r)+Math.abs(aiPos.c-chaosItem.c);
        const newD=Math.abs(a.pos.r-chaosItem.r)+Math.abs(a.pos.c-chaosItem.c);
        if(newD<oldD&&newD<=4)s+=(5-newD)*4; // näher heran lohnt, je näher desto mehr
        if(newD===0)s+=30; // direkt drauf (auch wenn schon oben abgefangen)
      }
      // Starke Cycling-Strafe: verhindert Hin-und-Her zwischen besuchten Feldern
      if(recent.length>=2){
        for(let i=1;i<Math.min(recent.length,5);i++){
          if(recent[i].r===a.pos.r&&recent[i].c===a.pos.c){
            s-=(6-i)*12; // -60 / -48 / -36 / -24 für i=1..4
            break;
          }
        }
      }
    }
    if(s>bestScore){bestScore=s;bestAction=a;}
  }

  // Fallback: vorwärts gehen wenn nichts gefunden
  if(!bestAction){
    const aiPath=bfsPath(aiPos,walls,aiGoal);
    if(aiPath&&aiPath.length>1)return{type:'move',pos:aiPath[1]};
    const fm=getMovesFrom(aiPos,humanPos,walls);
    if(fm.length)return{type:'move',pos:fm[0]};
    return null;
  }

  if(bestAction.type==='move')return{type:'move',pos:bestAction.pos};
  return{type:'barricade',walls:bestAction.walls};
}

// ── Replay ─────────────────────────────────────────────────
function applyMoves(moveList,step,map){
  const sp=startPos(map||'classic');
  let pA={...sp.A},pB={...sp.B};
  const walls=new Set();
  for(let i=0;i<Math.min(step,moveList.length);i++){
    const m=moveList[i];
    if(m.type==='emote')continue;
    if(m.type==='move'){if(m.p==='A')pA={r:m.r,c:m.c};else pB={r:m.r,c:m.c};}
    else if(m.type==='break')walls.delete(m.k);
    else walls.add(m.k);
  }
  return{pA,pB,walls};
}
function applyMovesWithOwner(moveList,step,map){
  const sp=startPos(map||'classic');
  let pA={...sp.A},pB={...sp.B};
  const walls=new Set();
  const wallOwner={};
  for(let i=0;i<Math.min(step,moveList.length);i++){
    const m=moveList[i];
    if(m.type==='emote')continue;
    if(m.type==='move'){if(m.p==='A')pA={r:m.r,c:m.c};else pB={r:m.r,c:m.c};}
    else if(m.type==='break'){walls.delete(m.k);delete wallOwner[m.k];}
    else{walls.add(m.k);wallOwner[m.k]=m.p||null;}
  }
  return{pA,pB,walls,wallOwner};
}
