# এই পরীক্ষাটি কী বোঝার জন্য

শির্ষা মিত্র / 2023521460130 / ১৬ নভেম্বর ২০২৬।

## আমরা এখন কী করছি

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

## ফলাফল কীভাবে বুঝবে

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
অবস্থা থাকবে। Complete and audited না হওয়া পর্যন্ত final result দাবি করবে না।
RESULTS.md, summary.csv, paired.json এবং audit.json তৈরি হলে সেগুলো পড়বে।

সর্বোচ্চ cloud budget $8 অনুমোদিত। Failed job বা cancelled laptop run নিজে থেকে
পুনরায় চালু হবে না। কোনো completed cloud command শুধু status দেখার জন্য আবার
চালাবে না।

