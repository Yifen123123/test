const q = document.getElementById('q');
const reveal = document.getElementById('reveal'); // 可能不存在（若你沒做顯示敏感資訊開關）
const cards = [...document.querySelectorAll('.card')];

function escapeRegExp(s){ return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'); }

const HIGHLIGHT_TARGETS = ['.title', '.summary']; // 只在標題與摘要做標亮
function setHighlights(card, tokens){
  HIGHLIGHT_TARGETS.forEach(sel=>{
    const el = card.querySelector(sel);
    if(!el) return;
    if(!el.dataset.orig){ el.dataset.orig = el.textContent; }
    const orig = el.dataset.orig;
    if(!tokens.length){
      el.innerHTML = orig;
      return;
    }
    let html = orig;
    tokens.forEach(t=>{
      const re = new RegExp(escapeRegExp(t), 'gi');
      html = html.replace(re, m => `<mark>${m}</mark>`);
    });
    el.innerHTML = html;
  });
}

let timer=null;
function applySearch(){
  const kw = (q?.value || '').trim().toLowerCase();
  const tokens = kw ? kw.split(/\s+/).filter(Boolean) : [];
  cards.forEach(c=>{
    // 每次動手現算，避免 reveal 切換後索引不同步
    const hay = c.innerText.toLowerCase();
    const ok = tokens.every(t => hay.includes(t)); // 多詞 AND
    c.style.display = ok ? '' : 'none';
    if(ok) setHighlights(c, tokens); else setHighlights(c, []);
  });
}

q?.addEventListener('input', ()=>{
  clearTimeout(timer);
  timer = setTimeout(applySearch, 120); // debounce
});

reveal?.addEventListener('change', applySearch); // 切換顯示敏感資訊時重新比對

// 初始套用
applySearch();
