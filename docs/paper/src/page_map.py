"""Extract heading -> page map from a PDF rendered from thesis.md (headings level 1-2 that appear as a line start)."""
import re, sys, json, pypdf
md, pdf, out = sys.argv[1:4]
heads = [(len(m.group(1)), m.group(2).strip()) for m in re.finditer(r"^(#{1,2})\s+(.*)$", open(md).read(), flags=re.M)]
heads = [(l, t) for l, t in heads if t not in ("Abstract",) or True]
r = pypdf.PdfReader(pdf); texts = [(p.extract_text() or "") for p in r.pages]
norm = [re.sub(r"\s+", " ", x) for x in texts]
toc_pages = {i for i, tt in enumerate(norm) if "Table of Contents" in tt or sum(1 for _, t in heads if re.sub(r"\s+", " ", t) in tt) >= 6}
res = []; start = 0
for l, t in heads:
    key = re.sub(r"\s+", " ", t)
    found = None
    for i in range(start, len(texts)):
        if i in toc_pages: continue
        if key in norm[i]: found = i + 1; break
    if found is None:
        for i in range(len(texts)):
            if i in toc_pages: continue
            if key in norm[i]: found = i + 1; break
    if found: start = max(start, found - 1)
    res.append({"level": l, "title": t, "page": found or "?"})
json.dump(res, open(out, "w"), indent=1); print(len(res), "headings; toc pages", sorted(i + 1 for i in toc_pages), "; unresolved", [x["title"] for x in res if x["page"] == "?"])
