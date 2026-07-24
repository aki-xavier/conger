#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#let template(doc) = {
  show heading: it => {
    if it.level == 1 {
      set align(center)
      it
    } else {
      it
    }
  }

  set heading(numbering: "1.1.1.1.1.")

  show heading.where(level: 1): set text(size: 15pt)
  show heading.where(level: 1): set block(above: 2em, below: 1em)
  show heading.where(level: 2): set text(size: 13pt)
  show heading.where(level: 3): set text(size: 11pt)

  show figure.caption: set text(size: 9pt)

  set math.equation(numbering: "(1)")

  show math.equation: set text(font: "TeX Gyre Termes Math")

  set text(
    font: (
      (name: "Times New Roman", covers: "latin-in-cjk"),
      "Noto Serif CJK SC",
    ),
    size: 10.5pt,
    lang: "zh",
    region: "CN",
  )

  show heading: set text(
    font: (
      (name: "Times New Roman", covers: "latin-in-cjk"),
      "Noto Sans CJK SC",
    ),
  )

  show: codly-init.with()
  codly(languages: codly-languages)

  set list(indent: 2em)
  set enum(indent: 2em)

  set par(
    first-line-indent: (
      amount: 2em,
      all: true,
    ),
    justify: true,
  )
  doc
}

#let tmp_page(doc) = {
  set page(
    numbering: "1",
    number-align: center + bottom,
  )
  doc
}

#let print_title(title, size: 18pt) = {
  set text(
    font: (
      (name: "Times New Roman", covers: "latin-in-cjk"),
      "Noto Sans CJK SC",
    ),
  )
  align(center, text(size)[
    *#title*
  ])
}

#let reportcoverpage(course_no, course_name, report_name, class, teacher, author, stu_no, sub_time) = {
  set text(
    font: (
      (name: "Times New Roman", covers: "latin-in-cjk"),
      "STHeiti",
    ),
  )
  table(
    columns: (1fr, 1fr),
    stroke: none,
    [课程编号：#underline(course_no) \ ],
    table(
      columns: (1fr, 1fr, 1fr),
      align: center,
      inset: 10pt,
      [得分], [教师签名], [批改日期],
      [\ ], [\ ], [\ ],
    ),
  )
  pad(top: 5em, bottom: 5em)[
    #align(center, [
      #image("sztu.png")
    ])
    #print_title("深圳技术大学实验报告", size: 26pt)
  ]
  table(
    columns: (1fr, 6fr, 1fr),
    stroke: none,
    [],
    [
      #set text(size: 14pt, weight: "bold")
      #table(
        columns: (1fr, 3fr),
        align: (right, center),
        gutter: 1em,
        stroke: (none, (bottom: 0.5pt + black)),
        [课程名称:], [#course_name],
        [报告名称:], [#report_name],
        [班级:], [#class],
        [指导教师：], [#teacher],
        [报告人：], [#author],
        [学号：], [#stu_no],
        [提交日期：], [#sub_time],
      )
    ],
    [],
  )

  pagebreak()
}

#let essaycoverpage(course_no, course_name, class, teacher, author, stu_no) = {
  set text(
    font: (
      (name: "Times New Roman", covers: "latin-in-cjk"),
      "STFangsong",
    ),
    size: 10pt,
  )

  table(
    columns: (1fr, 1.5fr, 1fr, 1.5fr, 1fr, 1.5fr, 0.8fr, 0.5fr),
    gutter: 0em,
    align: horizon,
    stroke: none,
    // justify: true,
    table.cell(align: right)[课程编号],
    table.cell(stroke: (bottom: 0.5pt))[#course_no],
    table.cell(align: right)[课程名称],
    table.cell(stroke: (bottom: 0.5pt))[#course_name],
    table.cell(align: right)[主讲教师],
    table.cell(stroke: (bottom: 0.5pt))[#teacher],
    table.cell(align: right)[评分],
    table.cell(stroke: (bottom: 0.5pt))[],
    table.cell(align: right)[学号],
    table.cell(stroke: (bottom: 0.5pt))[#stu_no],
    table.cell(align: right)[姓名],
    table.cell(stroke: (bottom: 0.5pt))[#author],
    table.cell(align: right)[专业年级],
    table.cell(stroke: (bottom: 0.5pt), colspan: 3)[#class],
  )

  block(stroke: 0.5pt, width: 100%, height: 100pt, inset: 1em)[教师评语：]
}

#let gradingtable() = {
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    align: center + horizon,
    stroke: 0.5pt,
    inset: 8pt,
    table.header(
      [*程序编写规范性、格式与质量*\ (15%)],
      [*程序编写现场完成情况*\ (40%)],
      [*程序编写结果*\ (40%)],
      [*思考题与实验总结*\ (5%)],
      [*总分*],
    ),
    [], [], [], [], [],
  )

  v(1em)

  align(right, [
    批阅教师签字： #box(width: 10em, stroke: (bottom: 0.5pt)) \

    日 #h(0.5em) 期： #box(width: 3em, stroke: (bottom: 0.5pt)) 年 #box(width: 3em, stroke: (bottom: 0.5pt)) 月 #box(width: 3em, stroke: (bottom: 0.5pt)) 日
  ])
}

#let gradingtable2() = {
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    align: center + horizon,
    stroke: 0.5pt,
    inset: 8pt,
    table.header(
      [*实验内容完整性*\ (30分)], [*实验现象和结果*\ (40分)], [*实验总结*\ (20分)], [*排版格式*\ (10分)], [*总分*]
    ),
    [], [], [], [], [],
  )
}

#let hl(body) = box(fill: yellow.lighten(70%), inset: (x: 0pt, y: 0pt), body)
