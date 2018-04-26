import time
import datetime

import pandas as pd

level = -1

class Watcher(object):

   def __init__(self):
       self.df = pd.DataFrame()
       self._kwargs = {}

   def __call__(self, *args, **kwargs):
       if args:
           self._name = args[0]
       self._kwargs.update(kwargs)
       return self

   def __enter__(self):
       global level
       level += 1
       # print()
       head = '          ' * level + '~~~~~~~~~~' * (4 - level)
       print('{}{:*^60}{}'.format(
           head,
           ' ' + str(self._name).upper() + ' ',
           head[::-1],
       ))
       # print('+' * 120)

       self._start_t = datetime.datetime.now()
       self._start_c = time.clock()

   def __exit__(self, *args, **kwargs):
       global level
       end_t = datetime.datetime.now()
       end_c = time.clock()

       delta_t = (end_t - self._start_t).total_seconds()
       delta_c = end_c - self._start_c

       d = dict(
           name=self._name,
           delta_t=delta_t,
       )
       d.update(self._kwargs)
       self._kwargs = {}

       self.df = pd.concat([self.df, pd.DataFrame([d])])
       # print('-' * 120)
       head = '          ' * level + '~~~~~~~~~~> '
       print(head + str(self).replace(
           '\n', '\n' + head
       ))
       level -= 1

   def __str__(self):
       df = self.df.copy().reset_index(drop=True)
       df['time%'] = df.delta_t / df.delta_t.sum()

       columns = ['name', 'delta_t',  'time%']
       columns += [s for s in df.columns if s not in columns]

       df = df[columns]

       def _perc(v):
           if v != v:
               return '--'
           else:
               return '{:6.2%}'.format(v)

       def _sep_int(v):
           v = str(int(v))
           def _f(v):
               count_full = len(v) // 3
               more = len(v) - count_full * 3
               if more:
                   yield v[:more]
               for i in range(count_full):
                   yield v[more + i * 3:more + (i + 1) * 3]
           return "'".join(list(_f(v)))

       def _int(v):
           if v != v:
               return '--'
           else:
               return _sep_int(v)

       def _default(v):
           if v != v:
               return '--'
           elif isinstance(v, float):
               return '{:.2f}'.format(v)
           else:
               return str(v)

       return(df.to_string(
           formatters={
               key: (
                   _perc if '%' in key else
                   _int if 'count' in key else
                   _default
               )
               for key in df.columns
           }
       ))
