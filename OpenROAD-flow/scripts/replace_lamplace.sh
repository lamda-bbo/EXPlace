#!/usr/bin/env sh
# 把 A、B 对应行合并，以 A 的坐标覆盖 B 的坐标，再输出 B 列

paste -d '\t' "$1" "$2" | sed -E '
  # 情况1：两边都有 -location
  s/^(.*-location[ \t]*\{)([^}]*)\}(.*)\t(.*-location[ \t]*\{)([^}]*)\}(.*)$/\4\2}\6/
  # 情况2：A 有 -location，B 没有（可选处理：在 B 末尾追加 -location）
  # 如需此功能，可取消下一行注释：
  # t
' | cut -f1