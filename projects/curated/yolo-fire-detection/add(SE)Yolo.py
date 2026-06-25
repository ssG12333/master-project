 if m in [Conv, GhostConv, Bottleneck, Bottleneck_cot,TransformerC3,GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3,seC3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, seC3]:
                args.insert(2, n)  # number of repeats
                n = 1