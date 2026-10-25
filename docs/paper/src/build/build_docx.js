// Markdown-subset -> .docx (docx v9). Usage: node build_docx.js thesis.md config.json out.docx
// Dialect: # ## ### #### headings; paragraphs; "- " bullets; "1. " numbered; pipe tables (first row header);
// ![caption](path) images; %PAGEBREAK; %TOC; %TABLE: caption (before a table); %EQ: text (centred);
// %ABSTRACT / %REFS / %ENDREFS section markers; inline **bold**, *italic*, `code`; a line "%LANDSCAPE" not supported.
const fs = require("fs"), path = require("path");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, TableOfContents, Footer, Header, PageNumber, NumberFormat, LevelFormat, TabStopType, VerticalAlign, SectionType } = D;

const [,, mdPath, cfgPath, outPath] = process.argv;
const md = fs.readFileSync(mdPath, "utf8"); const cfg = JSON.parse(fs.readFileSync(cfgPath, "utf8"));
const BODY_FONT = cfg.bodyFont || "Arial", HEAD_FONT = cfg.headingFont || "Times New Roman", BODY_SIZE = (cfg.bodySize || 11) * 2;
const TEXT_W = 12240 - 2 * 1440; // A4 width 11906 twips; we use US-Letter-like text width computed below instead
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1440, TW = PAGE_W - 2 * MARGIN; // A4, 1" margins, text width in twips

function inline(text, base = {}) {
  // returns TextRun[] with **bold**, *italic*, `code`, ^sup^ and ~sub~ handling
  const runs = []; const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\^[^^]+\^|~[^~]+~)/g; let last = 0, m;
  const push = (t, o) => { if (t) runs.push(new TextRun({ text: t, font: BODY_FONT, size: BODY_SIZE, ...base, ...o })); };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index)); const s = m[0];
    if (s.startsWith("**")) push(s.slice(2, -2), { bold: true });
    else if (s.startsWith("`")) push(s.slice(1, -1), { font: "Consolas", size: BODY_SIZE - 2 });
    else if (s.startsWith("^")) push(s.slice(1, -1), { superScript: true });
    else if (s.startsWith("~")) push(s.slice(1, -1), { subScript: true });
    else push(s.slice(1, -1), { italics: true });
    last = m.index + s.length;
  }
  push(text.slice(last)); return runs;
}

const para = (text, opts = {}, base = {}) => new Paragraph({ children: inline(text, base), spacing: { after: 120, line: 276 }, ...opts });
const children = []; let figN = 0, tabN = 0, tabPrefix = "", pendingTableCaption = null, inRefs = false, inAbstract = false;
const lines = md.split(/\r?\n/); let i = 0;
const listRef = "bullets", numRef = "numbers"; let numInstance = 0;

function flushParagraph(buf) { if (buf.length) { const t = buf.join(" ").trim(); if (t) children.push(para(t, inRefs ? { indent: { left: 720, hanging: 720 }, spacing: { after: 120, line: 276 } } : inAbstract ? { alignment: AlignmentType.JUSTIFIED } : {})); buf.length = 0; } }

function makeTable(rows) {
  const ncol = Math.max(...rows.map(r => r.length));
  // column widths: every column gets room for its longest word (absolute, from the font size), and the remaining width
  // is shared in proportion to each column's content length (mean and capped maximum cell length)
  const fsz = ncol >= 6 ? BODY_SIZE - 5 : BODY_SIZE - 3, cw = fsz * 5.6;   // approximate character width in DXA
  const clean = c => (c || "").replace(/[*`~^]/g, "").trim();
  const cols = Array.from({ length: ncol }, (_, ci) => {
    const cells = rows.map(r => clean(r[ci])); const body = cells.slice(1).map(c => c.length);
    const word = Math.min(20, Math.max(...cells.map(c => Math.max(0, ...c.split(/\s+/).map(w => w.length)))));
    const mean = body.reduce((x, y) => x + y, 0) / Math.max(1, body.length), mx = Math.min(Math.max(0, ...body), 80);
    return { min: (Math.max(word, 3) + 2) * cw, want: Math.max(0.55 * mean + 0.45 * mx, word) }; });
  let mins = cols.map(c => c.min), msum = mins.reduce((x, y) => x + y, 0);
  if (msum > TW) mins = mins.map(m => m * TW / msum), msum = TW;
  const extra = cols.map(c => c.want), esum = extra.reduce((x, y) => x + y, 0) || 1;
  const widths = mins.map((m, ci) => Math.floor(m + (TW - msum) * extra[ci] / esum));
  const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
  const trs = rows.map((r, ri) => new TableRow({ tableHeader: ri === 0, children: r.map((c, ci) => new TableCell({
    width: { size: widths[ci], type: WidthType.DXA }, borders: { top: border, bottom: border, left: border, right: border },
    shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "E7E6E6", color: "auto" } : undefined, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ children: inline(c.trim(), { size: fsz, bold: ri === 0 }), spacing: { after: 0, line: 240 }, alignment: AlignmentType.LEFT })] })) }));
  return new Table({ rows: trs, width: { size: TW, type: WidthType.DXA }, columnWidths: widths });
}

let buf = [];
while (i < lines.length) {
  const L = lines[i];
  if (/^\s*$/.test(L)) { flushParagraph(buf); i++; continue; }
  if (L.startsWith("%PAGEBREAK")) { flushParagraph(buf); children.push(new Paragraph({ children: [new PageBreak()] })); i++; continue; }
  if (L.startsWith("%TOC")) { flushParagraph(buf); children.push(new Paragraph({ children: [new TextRun({ text: "Table of Contents", font: HEAD_FONT, size: 32, bold: true })], spacing: { after: 240 } }));
    children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-3" })); i++; continue; }
  if (L.startsWith("%STATICTOC")) { flushParagraph(buf); const map = JSON.parse(fs.readFileSync(path.resolve(path.dirname(mdPath), L.slice(10).trim()), "utf8"));
    children.push(new Paragraph({ children: [new TextRun({ text: "Table of Contents", font: HEAD_FONT, size: 32, bold: true })], spacing: { after: 240 } }));
    for (const e of map) { const ind = (e.level - 1) * 360; children.push(new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: TW, leader: "dot" }], indent: { left: ind }, spacing: { after: 40, line: 264 },
      children: [new TextRun({ text: e.title, font: BODY_FONT, size: e.level === 1 ? BODY_SIZE : BODY_SIZE - 2, bold: e.level === 1 }), new TextRun({ text: "\t" + e.page, font: BODY_FONT, size: e.level === 1 ? BODY_SIZE : BODY_SIZE - 2 })] })); }
    i++; continue; }
  if (L.startsWith("%REFS")) { flushParagraph(buf); inRefs = true; i++; continue; }
  if (L.startsWith("%ENDREFS")) { flushParagraph(buf); inRefs = false; i++; continue; }
  if (L.startsWith("%ABSTRACT")) { flushParagraph(buf); inAbstract = true; i++; continue; }
  if (L.startsWith("%ENDABSTRACT")) { flushParagraph(buf); inAbstract = false; i++; continue; }
  if (L.startsWith("```")) { flushParagraph(buf); i++; const code = [];
    while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]); i++;
    code.forEach((c, k) => children.push(new Paragraph({ children: [new TextRun({ text: c || " ", font: "Courier New", size: BODY_SIZE - 4 })],
      shading: { type: ShadingType.CLEAR, fill: "F3F4F6", color: "auto" }, spacing: { before: k === 0 ? 80 : 0, after: k === code.length - 1 ? 160 : 0, line: 240 } })));
    continue; }
  if (L.startsWith("%TABLE:")) { flushParagraph(buf); pendingTableCaption = L.slice(7).trim(); i++; continue; }
  if (L.startsWith("%APPENDIX")) { flushParagraph(buf); tabPrefix = L.slice(9).trim(); tabN = 0; i++; continue; }
  if (L.startsWith("%EQ:")) { flushParagraph(buf); children.push(new Paragraph({ children: inline(L.slice(4).trim(), { italics: false }), alignment: AlignmentType.CENTER, spacing: { before: 60, after: 160 } })); i++; continue; }
  if (L.startsWith("%CENTER:")) { flushParagraph(buf); children.push(new Paragraph({ children: inline(L.slice(8).trim()), alignment: AlignmentType.CENTER, spacing: { after: 120 } })); i++; continue; }
  const h = L.match(/^(#{1,4})\s+(.*)$/);
  if (h) { flushParagraph(buf); const lvl = h[1].length; const text = h[2].trim();
    const hl = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3, HeadingLevel.HEADING_4][lvl - 1];
    children.push(new Paragraph({ heading: hl, children: [new TextRun({ text, font: HEAD_FONT, bold: true, size: [32, 28, 24, 22][lvl - 1], color: "000000" })],
      spacing: { before: lvl === 1 ? 360 : 240, after: 120 }, pageBreakBefore: lvl === 1 && cfg.h1PageBreak !== false && !/^(Abstract|Executive summary|Acknowledgements|Declaration)$/.test(text) ? true : false, keepNext: true })); i++; continue; }
  const img = L.match(/^!\[(.*)\]\((.*)\)\s*$/);
  if (img) { flushParagraph(buf); figN++; const p = path.resolve(path.dirname(mdPath), img[2]); const data = fs.readFileSync(p);
    const dims = pngSize(data); const maxW = 6.3 * 96, maxH = 8.2 * 96; let w = dims.w, hh = dims.h; const sc = Math.min(maxW / w, maxH / hh, 1); w = Math.round(w * sc); hh = Math.round(hh * sc);
    children.push(new Paragraph({ children: [new ImageRun({ type: "png", data, transformation: { width: w, height: hh } })], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true }));
    children.push(new Paragraph({ children: [new TextRun({ text: `Figure ${figN}. `, font: BODY_FONT, size: BODY_SIZE - 2, bold: true }), ...inline(img[1], { size: BODY_SIZE - 2 })], alignment: AlignmentType.LEFT, spacing: { after: 200, line: 240 } }));
    i++; continue; }
  if (/^\|/.test(L)) { flushParagraph(buf); const rows = [];
    while (i < lines.length && /^\|/.test(lines[i])) { const cells = lines[i].trim().replace(/^\|/, "").replace(/\|$/, "").split("|"); if (!/^\s*:?-{2,}/.test(cells[0])) rows.push(cells); i++; }
    tabN++; if (pendingTableCaption !== null) { children.push(new Paragraph({ children: [new TextRun({ text: `Table ${tabPrefix}${tabN}. `, font: BODY_FONT, size: BODY_SIZE - 2, bold: true }), ...inline(pendingTableCaption, { size: BODY_SIZE - 2 })], spacing: { before: 160, after: 80, line: 240 }, keepNext: true })); pendingTableCaption = null; }
    children.push(makeTable(rows)); children.push(new Paragraph({ children: [], spacing: { after: 120 } })); continue; }
  const b = L.match(/^\s*[-*]\s+(.*)$/);
  if (b) { flushParagraph(buf); children.push(new Paragraph({ children: inline(b[1]), numbering: { reference: listRef, level: 0 }, spacing: { after: 60, line: 276 } })); i++; continue; }
  const n = L.match(/^\s*(\d+)[.)]\s+(.*)$/);
  if (n) { flushParagraph(buf); if (n[1] === "1") numInstance++; children.push(new Paragraph({ children: inline(n[2]), numbering: { reference: numRef, level: 0, instance: numInstance }, spacing: { after: 60, line: 276 } })); i++; continue; }
  buf.push(L.trim()); i++;
}
flushParagraph(buf);

function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }

// title page
const tp = [];
const T = (t, o = {}) => tp.push(new Paragraph({ children: [new TextRun({ text: t, font: HEAD_FONT, ...o })], alignment: AlignmentType.CENTER, spacing: { after: o.after || 200 } }));
tp.push(new Paragraph({ spacing: { before: 2400 } }));
T(cfg.title, { size: 44, bold: true, after: 400 });
if (cfg.subtitle) T(cfg.subtitle, { size: 28, after: 900 });
T(cfg.author, { size: 26, after: 120 }); T(`Student number ${cfg.studentNumber}`, { size: 24, after: 600 });
T(cfg.degree, { size: 24, after: 120 }); T(cfg.school, { size: 24, after: 600 });
T(`Supervisor: ${cfg.supervisor}`, { size: 24, after: 120 }); T(cfg.date, { size: 24, after: 120 });
tp.push(new Paragraph({ children: [new PageBreak()] }));

const doc = new Document({
  creator: cfg.author, title: cfg.title,
  features: { updateFields: true },
  styles: { default: { document: { run: { font: BODY_FONT, size: BODY_SIZE } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: HEAD_FONT, size: 32, bold: true, color: "000000" }, paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: HEAD_FONT, size: 28, bold: true, color: "000000" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: HEAD_FONT, size: 24, bold: true, color: "000000" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
      { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: HEAD_FONT, size: 22, bold: true, italics: true, color: "000000" }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 3 } } ] },
  numbering: { config: [
    { reference: listRef, levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: numRef, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] } ] },
  sections: [
    { properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } }, titlePage: true },
      children: [...tp, ...children],
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: BODY_FONT, size: 20 })] })] }),
                 first: new Footer({ children: [new Paragraph({ children: [] })] }) } } ] });

Packer.toBuffer(doc).then(b => { fs.writeFileSync(outPath, b); console.log("wrote", outPath, b.length, "bytes;", figN, "figures,", tabN, "tables"); });
