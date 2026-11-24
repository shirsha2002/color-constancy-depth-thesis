# এই পরীক্ষাটি কী বোঝার জন্য

শির্ষা মিত্র / 2023521460130 / ১৬ নভেম্বর ২০২৬।

## পরীক্ষা শেষ: আসল ফলাফল

নয়টি নতুন model-এর প্রতিটি 10,000 epoch শেষ করেছে। পুরোনো ছয়টি reference
model-ও comparison-এ আছে। সব 15টি final checkpoint দিয়ে আবার prediction করা
হয়েছে এবং 3,240টি saved prediction-এর error স্বাধীনভাবে যাচাই করা হয়েছে।
Audit PASS। নতুন training বা automatic retry বাকি নেই।

| Input | Mean angular error |
|---|---:|
| Native geometry | 2.6135° |
| শুধু camera axes একই দিকে | 2.5471° |
| একই axes + median depth দিয়ে normalization | 2.7192° |
| একই axes-এর flat control | 2.8164° |
| আগের native flat control | 2.5061° |

কম error ভালো। কিন্তু শুধু গড়ের ছোট পার্থক্য দেখে উন্নতি নিশ্চিত বলা যায় না।
Normalization বনাম native এবং normalization বনাম matched flat - এই দুটি primary
comparison-এর interval-এই zero আছে। তাই এই পরীক্ষায় normalization-এর সুবিধা
প্রতিষ্ঠিত হয়নি। এটিকে “দুই পদ্ধতি নিশ্চিতভাবে সমান” বলাও ঠিক হবে না।
শুধু axes বদলের ছোট গড় লাভটিও অনিশ্চিত।

DIODE অংশে normalization করার পর mean error 2.6175° থেকে 3.3268° হয়েছে।
এই খারাপ ফলও রাখতে হবে; একই test দেখে ভালো ফল পাওয়ার জন্য transform বদলানো
ঠিক হবে না। Dataset ভুল বা scale-এর পার্থক্যই আগের ফলের কারণ - তা প্রমাণ হয়নি।

[English report](../output/pdf/Shirsha_Mitra_PCCC_Depth_Format_Update_2026_11_16.pdf),
[সব comparison](RESULTS.md), [audit](results/nyu_frame_v1/audit.json) এবং
[পরের কাজের প্রস্তাব](NEXT_STEPS.md) পড়ো। Reliability gate এখনো তৈরি হয়নি।

## আমরা কী পরীক্ষা করেছি

আমাদের বর্তমান PCCC task হলো রঙিন point cloud থেকে আলোর রং অনুমান করা।
প্রতিটি point-এ XYZ হলো অবস্থান এবং RGB হলো রং। Model-এর উত্তর একটি global
illuminant vector; তার angular error যত কম, তত ভালো। এটি depth map prediction
বা সম্পূর্ণ 3D reconstruction-এর accuracy পরীক্ষা নয়।

আগের cloud experiment-এ ETH3D-তে geometry ব্যবহার করে উপকার পাওয়া গেছে।
NYU+DIODE-তে একই রকম স্পষ্ট উপকার পাওয়া যায়নি। এই দুই উৎসের point cloud-এ
camera axes ও depth scale আলাদা পাওয়া গেছে। এই পার্থক্যই ফলাফলের কারণ কি না,
সেটি পরীক্ষা করাই বর্তমান কাজ। পার্থক্য পাওয়া মানেই dataset ভুল নয়।

## একই দৃশ্যের তিনটি নতুন representation

১. **camera_real:** camera axes একই দিকের করা হয়েছে। আসল scene scale রাখা হয়েছে।

২. **camera_unitdepth:** একই camera axes; প্রতিটি দৃশ্যের valid depth-এর median
দিয়ে পুরো XYZ ভাগ করা হয়েছে। ফলে median depth 1 হয়। কোন point অন্য point-এর
চেয়ে কত দূরে, সেই আপেক্ষিক সম্পর্ক থাকে; absolute scale আর থাকে না।

৩. **camera_flat:** একই camera axes, কিন্তু প্রত্যেক valid point-এর depth 1 করা
হয়েছে। ছবি বরাবর ray, RGB এবং missing-depth mask থাকে। আপেক্ষিক depth থাকে না।
এটিকে pure RGB-only model বলা যাবে না।

কোনো condition-এ RGB বা আসল আলোর রঙের label বদলানো হয়নি। Camera convention
বোঝার জন্য 16x16 grid-এর ordering এবং input geometry ব্যবহার করা হয়েছে; test
label বা model error ব্যবহার করা হয়নি।

## কেন নয়টি model

তিনটি নতুন condition × তিনটি official fold = নয়টি model। প্রতিটি fold-এ
বাকি দুই fold দিয়ে train করা হয় এবং নিজের fold-এ prediction করা হয়। সব fold
মিলিয়ে 324টি capture প্রত্যেকে একবার করে final evaluation-এ থাকে।

পুরোনো native এবং native-flat models আগে থেকেই train করা আছে। সেগুলো আবার train
না করে reuse করা হচ্ছে। এতে খরচ কমে এবং আগের reference অক্ষত থাকে।

সব নতুন model-এ released PCCC head, একই optimizer, learning rate, augmentation,
seed rule ও 10,000 epoch রাখা হয়েছে। নতুন reliability gate এখনো নেই।

## আগে নির্ধারিত ফল ব্যাখ্যা করার নিয়ম

নিচের সম্ভাবনাগুলো protocol-এর ব্যাখ্যার নিয়ম; কোনটি বাস্তবে হয়েছে, তা উপরের
completed ফলাফল থেকে বুঝতে হবে।

- **Normalized geometry native-এর চেয়ে ভালো, matched flat-এর চেয়ে ভালো নয়:**
  normalization উপকারী হতে পারে, কিন্তু relative depth-এর উপকার নিশ্চিত নয়।
- **Normalized geometry native এবং matched flat দুটির চেয়ে ভালো:** এই protocol-এ
  geometry ব্যবহার করার পক্ষে আরও ভালো evidence হতে পারে। Seed এবং নতুন real data
  দিয়ে আবার যাচাই দরকার।
- **ফল অপরিবর্তিত বা খারাপ:** sign/scale hypothesis সমর্থিত নাও হতে পারে। এটি
  গুরুত্বপূর্ণ ফল; ভালো দেখানোর জন্য test দেখে transformation বদলানো যাবে না।
- **শুধু camera axes বদলে লাভ:** orientation convention একটি ভূমিকা রাখতে পারে;
  সব লাভকে depth normalization-এর লাভ বলা যাবে না।

মূল comparison fixed শেষ epoch-এর model দিয়ে। Test fold দেখে বেছে নেওয়া best
checkpoint-এর ফল আলাদা optimistic reference হিসেবে থাকবে। দুই ধরনের ফল মেশানো
যাবে না। Error difference-এর interval-এ zero থাকলে নিশ্চিত improvement দাবি
করা উচিত নয়। এই interval training seed বা স্বাধীন room-এর সব অনিশ্চয়তা মাপে না।

## এরপর reliability-aware fusion কোথায় আসবে

বর্তমান পরীক্ষাটি data representation-এর diagnostic। পরের proposed algorithm-এ
colour ও geometry branch এবং একটি reliability gate থাকবে। Gate input থেকে
depth কতটা বিশ্বাসযোগ্য তা অনুমান করবে। Test-time আসল error, corruption label
বা ground-truth depth-এর গোপন সাহায্য gate পাবে না।

Ordinary fusion, missing-depth mask-এর সহজ control, learned gate এবং loss
ablations দরকার হবে। শুধু দুটি branch যোগ করাকে নতুন গবেষণা বলা যাবে না;
উপযুক্ত prior work ও controlled experiment দিয়ে নতুনত্ব দেখাতে হবে।

## এখন status কীভাবে দেখবে

এই command নতুন GPU চালু করে না:

```powershell
$env:PYTHONUTF8='1'
& .\.venv-modal\Scripts\python.exe phase7_depth_format/status.py
```

STATUS.md-তে human-readable অবস্থা, completion_status.json-এ local audit worker-এর
অবস্থা আছে। বর্তমান অবস্থা Complete and audited। RESULTS.md, summary.csv,
paired.json এবং audit.json তৈরি হয়েছে; সেগুলোতেই আসল ফল ও সীমাবদ্ধতা আছে।

এই শেষ হওয়া experiment-এর সর্বোচ্চ cloud budget $8 অনুমোদিত ছিল। এই অনুমতি
নতুন experiment-এর জন্য নয়। Failed job বা cancelled laptop run নিজে থেকে
পুনরায় চালু হবে না। কোনো completed cloud command শুধু status দেখার জন্য আবার
চালাবে না।

