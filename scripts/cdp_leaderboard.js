#!/usr/bin/env node
/** 从用户调试 Chrome 抓取 royaleapi.com/decks/leaderboard 第一页全部卡组并保存。
 *  用法: node scripts/cdp_leaderboard.js
 *  输出: docs/leaderboard_decks.json / docs/leaderboard_decks.txt
 */
const CDP = 'http://172.28.144.1:9222';
const fs = require('fs');
const URL = 'https://royaleapi.com/decks/leaderboard';
const OUT_JSON = '/mnt/e/clash-royale-simulator-main/docs/leaderboard_decks.json';
const OUT_TXT  = '/mnt/e/clash-royale-simulator-main/docs/leaderboard_decks.txt';

const EXTRACT_JS = `(() => {
  const rows = [...document.querySelectorAll('.deck_lb_row')];
  const out = [];
  for (const row of rows) {
    const playerA = row.querySelector('a.player_name');
    const clanA = row.querySelector('.clan_name a');
    const follow = row.querySelector('.follow_button[data-name]');
    const decklink = row.querySelector('a.decklink');
    // 依次找兄弟节点：deck_row（卡组图）与 lb_deck_info（数值）
    let sib = row.nextElementSibling;
    let deckRow = null, info = null;
    while (sib) {
      if (sib.classList && sib.classList.contains('deck_row')) deckRow = sib;
      if (sib.classList && sib.classList.contains('lb_deck_info')) { info = sib; break; }
      sib = sib.nextElementSibling;
    }
    const deckLink = deckRow ? deckRow.querySelector('a.deck_lb__deck_row') : null;
    const cards = (deckLink ? [...deckLink.querySelectorAll('img.deck_card')]
      .map(img => (img.className.match(/deck_card_key_([\\w-]+)/) || [])[1])
      .filter(Boolean) : []);
    const towerImg = info ? info.querySelector('.tower.item img.deck_card') : null;
    const tower = towerImg ? ((towerImg.className.match(/deck_card_key_([\\w-]+)/) || [])[1] || null) : null;
    const rank = info ? (info.querySelector('.rank.item') || {}).innerText?.trim() || null : null;
    const numOf = (iconSel) => {
      if (!info) return null;
      const icon = info.querySelector(iconSel);
      if (!icon) return null;
      const item = icon.closest('.item');
      const t = item ? item.innerText.trim() : '';
      const n = parseFloat(t);
      return isNaN(n) ? null : n;
    };
    const elixir = numOf('.elixir_icon');
    const cycle = numOf('.shortest_cycle_icon');
    const trophies = numOf('.rating_icon');
    let lastBattle = null;
    const ts = row.parentElement ? row.parentElement.querySelector('.deck_lb__timestamp_container .i18n_duration_short') : null;
    if (ts) lastBattle = ts.innerText.trim();
    let copyDeck = null;
    if (decklink) {
      const m = decklink.href.match(/copyDeck\\?deck=([^&]+)/);
      if (m) copyDeck = decodeURIComponent(m[1]);
    }
    out.push({
      rank: rank ? parseInt(rank, 10) : null,
      player: playerA ? playerA.innerText.trim() : null,
      player_tag: playerA ? (playerA.getAttribute('href') || '').replace('/player/', '') : null,
      clan: clanA ? clanA.innerText.trim() : null,
      clan_tag: clanA ? (clanA.getAttribute('href') || '').replace('/clan/', '') : null,
      tower,
      cards,
      deck_data_name: follow ? follow.getAttribute('data-name') : null,
      copy_deck: copyDeck,
      elixir,
      cycle,
      trophies,
      last_battle: lastBattle
    });
  }
  return JSON.stringify({ count: out.length, url: location.href, decks: out });
})()`;

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

  // 等 Cloudflare 通过 + SPA 渲染出卡组
  let ready = false;
  for (let i = 0; i < 40 && !ready; i++) {
    await new Promise(r => setTimeout(r, 1500));
    try {
      const r = await send('Runtime.evaluate', {
        expression: `JSON.stringify({title:document.title,ready:document.readyState,n:document.querySelectorAll('.deck_lb_row').length})`,
        returnByValue: true,
      });
      const v = JSON.parse(r.result.value);
      console.log(`[${i}] ready=${v.ready} n=${v.n} title=${v.title.slice(0,50)}`);
      if (v.ready === 'complete' && !/Just a moment|请稍候/i.test(v.title) && v.n > 0) ready = true;
    } catch (e) {}
  }
  await new Promise(r => setTimeout(r, 2000));

  const res = await send('Runtime.evaluate', { expression: EXTRACT_JS, returnByValue: true });
  const data = JSON.parse(res.result.value);
  console.log('extracted decks:', data.count, '| url:', data.url);

  fs.writeFileSync(OUT_JSON, JSON.stringify(data, null, 2), 'utf-8');

  // 人类可读文本
  const lines = [];
  lines.push(`RoyaleAPI Deck Leaderboard — ${data.url}`);
  lines.push(`抓取时间: ${new Date().toISOString()}`);
  lines.push(`卡组数: ${data.count}`);
  lines.push('='.repeat(100));
  for (const d of data.decks) {
    lines.push(`#${d.rank}  ${d.player} (${d.player_tag})  [${d.clan || '-'}${d.clan_tag ? ' / ' + d.clan_tag : ''}]`);
    lines.push(`   卡组: ${(d.cards || []).join(', ')}`);
    lines.push(`   塔: ${d.tower} | 圣水: ${d.elixir} | 最短循环: ${d.cycle} | 奖杯: ${d.trophies} | 最近天梯战: ${d.last_battle}`);
    if (d.copy_deck) lines.push(`   copyDeck: ${d.copy_deck}`);
    lines.push('');
  }
  fs.writeFileSync(OUT_TXT, lines.join('\n'), 'utf-8');
  console.log('saved:', OUT_JSON, OUT_TXT);

  ws.close();
  await fetch(`${CDP}/json/close/${tab.id}`).catch(() => {});
})().catch(e => { console.error('FAIL', e); process.exit(1); });
