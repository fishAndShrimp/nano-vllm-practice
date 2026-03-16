# Component range to Sequence
|Part|Component||Desc|
|-|-|-|-|
|Sequence.status|RequestQueue|Read Write|WAITING<=>RUNNING|
|Sequence.status|Scheduler|Read Write|RUNNING=>FINISHED|
|Sequence|Scheduler|Read Write||
|Sequence|KVCacheManager|Read Only||
