#!/usr/bin/env node
/** Probe royaleapi deck leaderboard via user's debug Chrome (CDP) */
const CDP = 'http://172.28.144.1:9222';
const fs = require('fs');
const URL = process.argv[2] || 'https://royaleapi.com/decks/leaderboard';

(async () => {
  const tab = await (await fetch(`${CDP}/json/new`, { method: 'PUT' })).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  let id = 0; const pending = new Map(); const waiters = [];
  const send = (method, params = {}) => new Promise((res) => {
    const i = ++id; pending.set(i, res);
    ws.send(JSON.stringify({ id: i, method, params }));
  });
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); return; }
    if (m.method === 'Page.loadEventFired') waiters.splice(0).forEach(f => f());
  };
  await new Promise(r => { ws.onopen = r; });
  await send('Page.enable'); await send('Runtime.enable');

  const loaded = new Promise(r => waiters.push(r));
  await send('Page.navigate', { url: URL });
  await Promise.race([loaded, new Promise(r => setTimeout(r, 20000))]);

  // Wait for Cloudflare / SPA hydration
  for (let i = 0; i < 40; i++) {
    await new Promise(r => setTimeout(r, 1500));
    try {
      const r = await send('Runtime.evaluate', {
        expression: `JSON.stringify({title:document.title,ready:document.readyState,href:location.href,textLen:(document.body?document.body.innerText.length:0)})`,
        returnByValue: true,
      });
      const v = JSON.parse(r.result.value);
      console.log(`[${i}] ready=${v.ready} title=${v.title.slice(0,60)} href=${v.href.slice(0,70)} textLen=${v.textLen}`);
      const done = v.ready === 'complete' && !/Just a moment/i.test(v.title) && v.textLen > 500;
      if (done) break;
    } catch (e) {}
  }
  await new Promise(r => setTimeout(r, 3000));

  const res = await send('Runtime.evaluate', {
    expression: `JSON.stringify({href:location.href,title:document.title,text:(document.body?document.body.innerText:''),html:(document.body?document.body.outerHTML:'')})`,
    returnByValue: true,
  });
  const v = JSON.parse(res.result.value);
  fs.writeFileSync('/mnt/e/clash-royale-simulator-main/docs/_leaderboard_probe.txt', v.text, 'utf-8');
  fs.writeFileSync('/mnt/e/clash-royale-simulator-main/docs/_leaderboard_probe.html', v.html, 'utf-8');
  console.log('saved text len', v.text.length, 'html len', v.html.length);
  ws.close();
  await fetch(`${CDP}/json/close/${tab.id}`).catch(() => {});
})();
