with cc2:
                st.subheader("🌌 Equity Scatter")
                cmode = st.radio("Scatter", ["Mode A", "Mode B"], horizontal=True, label_visibility="collapsed")
                st.caption("Mode A: Eq vs Quality / Mode B: Eq vs Nut Eq")
                
                # Zoom Checkbox (復活)
                use_auto_zoom = st.checkbox("🔍 Auto Zoom", value=True)

                @st.cache_data
                def get_bg(df): return df.sample(3000, random_state=42).copy()
                bg = get_bg(df_plo)

                fig2, ax2 = plt.subplots(figsize=(5, 4))
                def gxy(d, m): return d["equity"], (d["nut_quality"] if "Mode A" in m else d["nut_equity"])
                
                bx, by = gxy(bg, cmode)
                mx, my = gxy(pd.DataFrame([row]), cmode); mx, my = mx.iloc[0], my.iloc[0]
                cbg = bg["nut_quality"] if "Mode A" in cmode else (1.0-(bx-by))
                
                # Background
                ax2.scatter(bx, by, c=cbg, cmap="coolwarm_r", s=10, alpha=0.1, label='Others')
                if "Mode B" in cmode: ax2.plot([0,1],[0,1], ls="--", c="gray", alpha=0.5)

                # Zoom Range Initialization
                xmin, xmax, ymin, ymax = mx, mx, my, my
                focused = False

                # Filtered (Gold)
                if filtered_df is not None and not filtered_df.empty:
                    fdf = filtered_df.sample(2000, random_state=42) if len(filtered_df)>2000 else filtered_df
                    fx, fy = gxy(fdf, cmode)
                    ax2.scatter(fx, fy, fc='none', ec='gold', s=30, label='Filtered')
                    # Update Range
                    xmin, xmax = min(xmin, fx.min()), max(xmax, fx.max())
                    ymin, ymax = min(ymin, fy.min()), max(ymax, fy.max())
                    focused = True
                
                # Highlight Groups (1-3)
                for i, tags in enumerate([hl_tags_1, hl_tags_2, hl_tags_3]):
                    if tags:
                        src = filtered_df if filtered_df is not None else df_plo
                        ht = set(tags)
                        mask = src["tags"].apply(lambda t: ht.issubset(set(t)))
                        hdf = src[mask]
                        if not hdf.empty:
                            hdf_s = hdf.sample(2000, random_state=42) if len(hdf)>2000 else hdf
                            hx, hy = gxy(hdf_s, cmode)
                            
                            colors = ['crimson', 'dodgerblue', 'limegreen']
                            grp_lbl = ["Grp1(Red)", "Grp2(Blue)", "Grp3(Grn)"]
                            label_text = f"{grp_lbl[i]}: {','.join(tags)[:10]}.."
                            
                            ax2.scatter(hx, hy, fc='none', ec=colors[i], s=50, lw=1.5, label=label_text)
                            
                            # Update Range
                            xmin, xmax = min(xmin, hx.min()), max(xmax, hx.max())
                            ymin, ymax = min(ymin, hy.min()), max(ymax, hy.max())
                            focused = True

                # You
                ax2.scatter(mx, my, c='black', s=150, marker='*', ec='white', zorder=10, label='You')
                
                # Auto Zoom Logic (ここを追加)
                if use_auto_zoom:
                    if not focused: 
                        xmin, xmax, ymin, ymax = bx.min(), bx.max(), by.min(), by.max()
                    
                    # マージンと最小幅の確保
                    if xmax == xmin: xmin -= 0.1; xmax += 0.1
                    if ymax == ymin: ymin -= 0.1; ymax += 0.1
                    
                    x_span = xmax - xmin
                    y_span = ymax - ymin
                    if x_span < 0.15: d=(0.15-x_span)/2; xmin-=d; xmax+=d
                    if y_span < 0.15: d=(0.15-y_span)/2; ymin-=d; ymax+=d

                    margin = 0.05
                    ax2.set_xlim(max(0, xmin-margin), min(1, xmax+margin))
                    ax2.set_ylim(max(0, ymin-margin), min(1, ymax+margin))
                else:
                    ax2.set_xlim(0, 1.05)
                    ax2.set_ylim(0, 1.05)

                ax2.grid(True, ls='--', alpha=0.3)
                ax2.legend(fontsize=8, loc='upper left')
                st.pyplot(fig2)
