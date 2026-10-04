# Broken case: theory case with a grade

Free text for people.

```toml
[identity]
id = "broken_theory_grade"
title = "Broken theory case with a grade"
entry_version = 1

[source]
citation = "Some closed-form source"
location = "Closed form"
type = "theory"
grade = "C"
synthetic = true

[geometry]
span_m = 6.0
chord_m = 1.0

[conditions]
mach = 0.1
reynolds = 1000000.0
beta_deg = 0.0
corrections = "none stated"

[[results]]
quantity = "CL"
alpha_deg = [0.0, 4.0]
values = [0.0, 0.4]
extraction = "table"

[quality]
notes = "A theory case with a grade on purpose."
```
