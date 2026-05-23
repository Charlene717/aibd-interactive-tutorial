(function(){
  const SK='ai_biobd_lang';
  let lang=localStorage.getItem(SK)||'en';

  function apply(l){
    lang=l; localStorage.setItem(SK,l);
    document.documentElement.lang=l==='zh'?'zh-Hant':'en';
    document.querySelectorAll('[data-lang]').forEach(el=>{
      el.style.display=el.dataset.lang===l?'':'none';
    });
    document.querySelectorAll('[data-zh][data-en]').forEach(el=>{
      el.textContent=el.dataset[l];
    });
    const btn=document.getElementById('langToggle');
    if(btn) btn.textContent=l==='zh'?'EN':'中文';
    document.dispatchEvent(new CustomEvent('langchange',{detail:{lang:l}}));
  }

  function toggle(){ apply(lang==='zh'?'en':'zh'); }
  function get(){ return lang; }

  document.addEventListener('DOMContentLoaded',()=>{
    const nav=document.querySelector('.top-nav-inner');
    if(nav && !document.getElementById('langToggle')){
      const btn=document.createElement('button');
      btn.id='langToggle'; btn.className='lang-toggle';
      btn.textContent=lang==='zh'?'EN':'中文';
      btn.onclick=toggle; nav.appendChild(btn);
    }
    apply(lang);
  });

  window.I18n={apply,toggle,get};
})();

// Reading progress bar (shared)
document.addEventListener('DOMContentLoaded',()=>{
  const bar=document.getElementById('progressBar');
  if(!bar) return;
  window.addEventListener('scroll',()=>{
    const h=document.documentElement;
    const pct=(h.scrollTop)/(h.scrollHeight-h.clientHeight)*100;
    bar.style.width=Math.min(100,Math.max(0,pct))+'%';
  });
});

// Code tab switching (shared)
document.addEventListener('click',(e)=>{
  const btn=e.target.closest('.code-tab-btn');
  if(!btn) return;
  const wrap=btn.closest('.code-tabs'); if(!wrap) return;
  wrap.querySelectorAll('.code-tab-btn').forEach(b=>b.classList.remove('active'));
  wrap.querySelectorAll('.code-tab-content').forEach(c=>c.classList.remove('active'));
  btn.classList.add('active');
  const tab=btn.dataset.tab;
  const tgt=wrap.querySelector('.code-tab-content[data-tab="'+tab+'"]');
  if(tgt) tgt.classList.add('active');
});

// Accordion (shared)
document.addEventListener('click',(e)=>{
  const btn=e.target.closest('.accordion-btn');
  if(!btn) return;
  btn.classList.toggle('open');
  const body=btn.nextElementSibling;
  if(body && body.classList.contains('accordion-body')) body.classList.toggle('open');
});

// Quiz answer feedback (shared)
document.addEventListener('click',(e)=>{
  const opt=e.target.closest('.quiz-opt');
  if(!opt) return;
  const q=opt.closest('.quiz-q'); if(!q) return;
  if(q.dataset.answered) return;
  q.dataset.answered='1';
  q.querySelectorAll('.quiz-opt').forEach(o=>{
    if(o.dataset.correct==='1') o.classList.add('correct');
  });
  if(opt.dataset.correct!=='1') opt.classList.add('wrong');
  const fb=q.querySelector('.quiz-feedback');
  if(fb){fb.classList.add('show'); fb.classList.add(opt.dataset.correct==='1'?'correct-fb':'wrong-fb');}
});
