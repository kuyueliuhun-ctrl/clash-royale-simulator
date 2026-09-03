#!/usr/bin/env node
/** CDP 页面文本抽取器 v2：Page.navigate 驱动（兼容 Chrome 152）
 *  用法: node scripts/cdp_extract.js <url1> <url2> ...
 *  输出: /tmp/cdp_page_<n>.txt
 */
const CDP = 'http://172.28.144.1:9222';
const fs = require('fs');

async function extractPage(url, idx) {
  const tab = await (await fetch(`${CDP}/json/new`, { method: 'PUT' })).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();
  const waiters = [];
  const send = (method, params = {}) => new Promise((res) => {
    const i = ++id;
    pending.set(i, res);
    ws.send(JSON.stringify({ id: i, method, params }));
  });
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); return; }
    if (m.method === 'Page.loadEventFired') { waiters.splice(0).forEach((f) => f()); }
  };
  await new Promise((r) => { ws.onopen = r; });
  await send('Page.enable');
  await send('Runtime.enable');

  const loaded = new Promise((r) => waiters.push(r));
  await send('Page.navigate', { url });
  const nav = await Promise.race([loaded, new Promise((r) => setTimeout(r, 20000))]);

  // 轮询 readyState 到 complete，再给 SPA 水合缓冲
  for (let i = 0; i < 20; i++) {
    await new Promise((r) => setTimeout(r, 1200));
    try {
      const rd = await send('Runtime.evaluate', { expression: 'document.readyState', returnByValue: true });
      if (rd && rd.result && rd.result.value === 'complete') break;
    } catch (e) { /* 跳转期忽略 */ }
  }
  await new Promise((r) => setTimeout(r, 3000));

  const info = await send('Runtime.evaluate', {
    expression: 'JSON.stringify({href:location.href,title:document.title,textLen:(document.body.innerText||"").length})',
    returnByValue: true,
  });
  let text = '';
  try {
    const t = await send('Runtime.evaluate', { expression: 'document.body.innerText', returnByValue: true });
    text = (t.result && t.result.value) || '';
  } catch (e) { text = 'EVAL_FAIL: ' + e.message; }
  ws.close();
  await fetch(`${CDP}/json/close/${tab.id}`).catch(() => {});
  const out = `/mnt/e/clash-royale-simulator-main/docs/_hero_${idx}.txt`;
  fs.writeFileSync(out, text, 'utf-8');
  console.log(`[${idx}] ${url}\n    -> ${out} (${text.length} chars) | ${info.result.value}`);
}

(async () => {
  const urls = process.argv.slice(2);
  for (let i = 0; i < urls.length; i++) {
    try { await extractPage(urls[i], i + 1); } catch (e) { console.log(`[${i + 1}] ${urls[i]} 失败: ${e.message}`); }
  }
})();
