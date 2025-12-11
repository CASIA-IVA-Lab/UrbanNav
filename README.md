<br>
<p align="center">

<h1 align="center"><strong>UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories</strong></h1>
  <p align="center"><span><a href=""></a></span>
              <a>Yanghong Mei<sup>*1,5</sup>,</a>
              <a>Yirong Yang<sup>*2</sup>,</a>
             <a>Longteng Guo<sup>†1</sup>,</a>
              <a>Qunbo Wang<sup>3</sup>,</a>
              <a>Ming-Ming Yu<sup>2</sup>,</a> <br>
              <a>Xingjian He<sup>1</sup>,</a>
              <a>Wenjun Wu<sup>2,4</sup>,</a>
              <a>Jing Liu<sup>1,5</sup>,</a>
    <br>
    <sup>1</sup>Institute of Automation, Chinese Academy of Sciences <br>
    <sup>2</sup>Beihang University <br>
    <sup>3</sup>Beijing Jiaotong University <br> 
    <sup>4</sup>Hangzhou International Innovation Institute <br>
    <sup>5</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences
    <br>
  </p>
    
<p align="center">
  <a href="https://arxiv.org/abs/2512.09607" target="_blank">
    <img src="https://img.shields.io/badge/ArXiv-2512.09607-red">
  </a>
  <a href="https://github.com/Vigar0108M/UrbanNav" target="_blank">
    <img src="https://img.shields.io/badge/Project-UrbanNav-blue">
  </a>
<a href="https://github.com/Vigar0108M/UrbanNav" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green">
</a>
</p>


![](src/UrabnNav-Demo.gif)
**UrbanNav** is a large-scale urban navigation dataset automatically constructed using Qwen2.5-VL based on web data, comprising 47k trajectories and 3M language instructions. Based on this dataset, we train language-conditioned navigation models designed for complex and dynamic outdoor scenarios. Our approach is deployed on a four-wheel differential-drive robot and validated in challenging and diverse outdoor urban environments.

## Abstract
Navigating complex urban environments using natural language instructions poses significant challenges for embodied agents, including noisy language instructions, ambiguous spatial references, diverse landmarks, and dynamic street scenes. Current visual navigation methods are typically limited to simulated or off-street environments, and often rely on precise goal formats, such as specific coordinates or images. This limits their effectiveness for autonomous agents like last-mile delivery robots navigating unfamiliar cities. To address these limitations, we introduce UrbanNav, a scalable framework that trains embodied agents to follow free-form language instructions in diverse urban settings. Leveraging web-scale city walking videos, we develop an scalable annotation pipeline that aligns human navigation trajectories with language instructions grounded in real-world landmarks. UrbanNav encompasses over 1,500 hours of navigation data and 3 million instruction-trajectory-landmark triplets, capturing a wide range of urban scenarios. Our model learns robust navigation policies to tackle complex urban scenarios, demonstrating superior spatial reasoning, robustness to noisy instructions, and generalization to unseen urban settings. Experimental results show that UrbanNav significantly outperforms existing methods, highlighting the potential of large-scale web video data to enable language-guided, real-world urban navigation for embodied agents.


![](src/overview.png)


## UrbanNav Dataset
The UrbanNav dataset will be publicly released by the end of 2025.


## 🌟 Citation

If you find this repository or our paper useful, please consider **starring** this repository and **citing** our paper:
```bibtex
@misc{mei2025urbannavlearninglanguageguidedurban,
      title={UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories}, 
      author={Yanghong Mei and Yirong Yang and Longteng Guo and Qunbo Wang and Ming-Ming Yu and Xingjian He and Wenjun Wu and Jing Liu},
      year={2025},
      eprint={2512.09607},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.09607}, 
}
```
