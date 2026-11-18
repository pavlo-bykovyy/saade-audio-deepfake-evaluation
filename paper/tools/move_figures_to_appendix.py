from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Cm
from collections import OrderedDict
from pathlib import Path
import re

SRC=Path('/mnt/data/saade_results_paper/SAADE_Results_MDPI_AI_35_figures_UK.docx')
DST=Path('/mnt/data/saade_results_paper/SAADE_Results_UK_Main_and_Appendix_A.docx')
keep={3,4,5,6,8,9,10,11,12,14}
d=Document(SRC)
body=d.element.body
blocks=list(body.iterchildren())
cap_rx=re.compile(r'^Рисунок\s+(\d+)\.\s*(.*)', re.DOTALL)
figs={}
for i, block in enumerate(blocks):
    if block.tag != qn('w:p'): continue
    p=Paragraph(block,d)
    match=cap_rx.match(p.text)
    if not match: continue
    num=int(match.group(1))
    prev=blocks[i-1]
    assert prev.xpath('.//w:drawing'), f'No image before fig {num}'
    figs[num]=(prev,block,match.group(2))
assert sorted(figs)==list(range(1,36)),sorted(figs)
assert len(d.inline_shapes)==35

# Locate original appendix introduction and remove; moving outside doc won't drop image rels.
for block in blocks:
    if block.tag!=qn('w:p'):continue
    p=Paragraph(block,d)
    if p.text.startswith('Додаткові візуальні результати — галерея рисунків') or p.text.startswith('Наведені нижче рисунки охоплюють усі оригінальні'):
        body.remove(block)

extras=[n for n in range(1,36) if n not in keep]
new_main={n:i+1 for i,n in enumerate(sorted(keep))}
new_app={n:f'А{i+1}' for i,n in enumerate(extras)}
for n in extras:
    img,cap,_=figs[n]
    body.remove(img)
    body.remove(cap)

# Update captions for preserved main figures
for n in sorted(keep):
    img_el, cap_el, desc=figs[n]
    cap=Paragraph(cap_el,d)
    cap.text=f'Рисунок {new_main[n]}. {desc}'
    cap.style='Figure Caption'
    cap.paragraph_format.keep_together=True
    cap.paragraph_format.space_after=Pt(8)
    p=Paragraph(img_el,d)
    p.paragraph_format.keep_with_next=True
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER

# Remove purely blank paragraphs that now occur around former images, without destroying real content.
for el in list(body.iterchildren()):
    if el.tag!=qn('w:p'): continue
    p=Paragraph(el,d)
    if p.text.strip() or el.xpath('.//w:drawing') or el.xpath('.//w:br'): continue
    # remove if there is an adjacent blank or a table or a heading around it
    prev_el=el.getprevious(); next_el=el.getnext()
    if prev_el is not None and (prev_el.tag==qn('w:tbl') or (prev_el.tag==qn('w:p') and Paragraph(prev_el,d).text.startswith('Рисунок'))):
        body.remove(el)

# Cross-references to supplementary material at scientifically appropriate locations.
anchors={
 '3.1. Обсяг оцінювання, аудит набору даних і повнота експерименту': 'Деталізовані графіки кількості записів за джерелами та розподілу зіставлених пар наведено в додатку А (рис. А1–А2).',
 '3.2. Основні результати на зіставленій валідаційній вибірці': 'Розгорнуті криві BA за п’ятьма рівнями кожного акустичного спотворення на validation, а також оригінальні графіки експерименту наведено в додатку А (рис. А5–А12).',
 '3.3. Одноразове тестування прихованої вибірки та перенесення на новий синтетичний голос': 'Додатковий графік змін ASRI між двома голосовими умовами наведено в додатку А (рис. А3), а деталізоване порівняння чистого і спотвореного аудіо — на рис. А13.',
 '3.4. Сімейства акустичних спотворень виявляють різноспрямовані профілі ризику': 'Повний аналіз ROC-AUC, окремих рівнів спотворення та профілів TPR/TNR представлено в додатку А (рис. А14–А25).',
 '3.5. Вплив передобробки аудіо відповідно до еталонного протоколу': 'Частку змінених рішень та оригінальний графік порівняння процедур передобробки винесено до додатка А (рис. А4 і А12).',
 '3.6. Канонічний AUA: повне сценарне тестування за джерелами': 'Оригінальний графік класоспецифічних результатів за джерелами наведено в додатку А (рис. А6).',
}
# Replace incorrect appendix numbering reference ranges below once reordered.
# Actual old-to-new mapping:
# A1 1, A2 2, A3 7, A4 13, A5 15 A6 16 A7 17 A8 18 A9 19 A10 20 A11 21 A12 22
# A13 23, A14 24 ... A25 35
for el in list(body.iterchildren()):
    if el.tag != qn('w:p'): continue
    p=Paragraph(el,d)
    if p.text.strip() in anchors:
        note=Paragraph(body.insert if False else el,d) if False else None
        from docx.oxml import OxmlElement
        node=OxmlElement('w:p')
        el.addnext(node)
        note=Paragraph(node,d)
        note.style='Normal'
        rr=note.add_run(anchors[p.text.strip()]); rr.italic=True
        note.paragraph_format.space_after=Pt(7)

h=d.add_heading('Додаток А. Додаткові візуальні результати',level=1)
h.paragraph_format.page_break_before=True
h.paragraph_format.keep_with_next=True
p=d.add_paragraph('У додатку подано 25 додаткових рисунків, переміщених із основного тексту для збереження його компактності. Кожний рисунок містить оригінальні дані експериментів v9.1/v9.2 або їхню повторну візуалізацію за архівними таблицями. Позначення А1–А25 використано для окремої нумерації додатка; рисунки 1–10 залишено в основному тексті.')
p.paragraph_format.space_after=Pt(10)

sections=OrderedDict([
 ('А.1. Склад корпусу та розподіл зіставлених пар', [1,2]),
 ('А.2. Додатковий аналіз перенесення та передобробки',[7,13]),
 ('А.3. Оригінальні діаграми validation та контроль аудіо',[15,16,17,18,19,20,21,22]),
 ('А.4. Розгорнуті сценарні результати validation та hidden',[23,24,25,26,27,28,29,30,31,32,33,34,35])
])
for heading, numbers in sections.items():
    sub=d.add_heading(heading,level=2)
    sub.paragraph_format.keep_with_next=True
    for n in numbers:
        img_el,cap_el,desc=figs[n]
        assert img_el.getparent() is None and cap_el.getparent() is None
        # Append before sectPr, not behind it
        body.insert(len(body)-1,img_el)
        body.insert(len(body)-1,cap_el)
        img_p=Paragraph(img_el,d)
        img_p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        img_p.paragraph_format.keep_with_next=True
        img_p.paragraph_format.keep_together=True
        cap=Paragraph(cap_el,d)
        cap.text=f'Рисунок {new_app[n]}. {desc}'
        cap.style='Figure Caption'
        cap.paragraph_format.keep_together=True
        cap.paragraph_format.space_after=Pt(12)

# Correct ranges in anchors for accurate addendum mapping:
# А.4–А.11 originals: 13,15,...21; А12 old22; A13 old23; A14..A25 old24-35.
# subsection3.4 A14..A25 etc valid.
# Insert a footer note at beginning of appendix to avoid conflating related/duplicated originals.

DST.parent.mkdir(parents=True,exist_ok=True)
d.save(DST)
new=Document(DST)
assert len(new.inline_shapes)==35
assert len(new.tables)==4
caps=[p.text for p in new.paragraphs if p.text.startswith('Рисунок ')]
assert len(caps)==35
assert len([x for x in caps if re.match(r'Рисунок \d+\.',x)])==10
assert len([x for x in caps if re.match(r'Рисунок А\d+\.',x)])==25
print('SAVED',DST,DST.stat().st_size,'captions',len(caps),'main',len(keep),'appendix',len(extras))
print('MAIN:',[(n,new_main[n]) for n in sorted(keep)])
print('APP:',[(n,new_app[n]) for n in extras])
