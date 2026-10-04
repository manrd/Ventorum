# Broken case: lists of different lengths

Free text for people.

```toml
[identity]
id = "broken_length_mismatch"
title = "Broken case with ragged lists"
entry_version = 1

[source]
citation = "Some source"
location = "Table 1"
type = "experiment"
grade = "B"
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
alpha_deg = [0.0, 2.0, 4.0]
values = [0.0, 0.4]
extraction = "table"

[quality]
notes = "Two values for three angles on purpose."
```
