#!/usr/bin/env python

from numpy import cumsum

def splitList(inputlist: list, batch_num: int = 1) -> list:
    items: int = len(inputlist)
    step: int = items//batch_num

    # split array in batches
    batches: list = []
    if (step > 0):
        for k in range(0, items, step):
            batches.append(inputlist[k:k+step])

        print(f"Generated {len(batches)} batches of size {step}")
        print(cumsum([len(x) for x in batches]))
        return batches
    else:
      print(f"Refusing to split: Cannot prepare batches of {step} length")
      return inputlist
