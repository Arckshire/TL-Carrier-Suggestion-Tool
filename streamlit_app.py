import streamlit as st
import pandas as pd
import numpy as np
import io

# ------------------------
# Utility Functions
# ------------------------
def parse_do_not_replace(value):
    if value is None:
        return []
    if str(value).strip().lower() == 'na':
        return []
    return [x.strip() for x in str(value).split(',') if str(x).strip()]

def compare_metrics(sugg_val, replace_val, direction):
    try:
        if pd.isnull(sugg_val) or pd.isnull(replace_val):
            return False
        if direction == 'higher':
            return sugg_val > replace_val   # strictly better (no ties)
        else:
            return sugg_val < replace_val
    except Exception:
        return False

def coerce_numeric(series: pd.Series) -> pd.Series:
    # strip %, commas, blanks, hyphens then coerce
    return (
        series.astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '', regex=False)
        .replace({'': np.nan, 'nan': np.nan, '-': np.nan, ' ': np.nan})
        .astype(float)
    )

# Metric definitions
metric_info = {
    "Tracking %": {"col": "Tracking %", "direction": "higher"},
    "Avg Ping Frequency Mins": {"col": "Avg Ping Frequency Mins", "direction": "lower"},
    "Milestone Completeness %": {"col": "Milestone Completeness Percent", "direction": "higher"},
    "Origin Arrival %": {"col": "Origin Arrival Milestones Percent", "direction": "higher"},
    "Origin Departure %": {"col": "Origin Departure Milestones Percent", "direction": "higher"},
    "Destination Arrival %": {"col": "Destination Arrival Milestones Percent", "direction": "higher"},
    "Destination Departure %": {"col": "Destination Departure Milestones Percent", "direction": "higher"},
    "Pickup Arrival <30 Min %": {"col": "Pickup Arrival Within 30 Min Percent", "direction": "higher"},
    "Dropoff Arrival <30 Min %": {"col": "Dropoff Arrival Within 30 Min Percent", "direction": "higher"},
}

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Carrier Suggestions (TL)", page_icon="🚚", layout="wide")
st.title("🚚 Carrier Suggestion Tool for Truckload Lanes")

uploaded_file = st.file_uploader("Upload the CSV file", type=["csv"])

if uploaded_file:
    with st.spinner("Reading file..."):
        df = pd.read_csv(uploaded_file)

    required_columns = [
        'Pickup Location', 'Dropoff Location', 'Type', 'Carrier Name', 'Shipment Volume',
        'Tracking %', 'Avg Ping Frequency Mins', 'Milestone Completeness Percent',
        'Origin Arrival Milestones Percent', 'Origin Departure Milestones Percent',
        'Destination Arrival Milestones Percent', 'Destination Departure Milestones Percent',
        'Pickup Arrival Within 30 Min Percent', 'Dropoff Arrival Within 30 Min Percent'
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"The uploaded file is missing these required columns: {', '.join(missing_cols)}")
        st.stop()

    # Clean numeric columns so weird inputs don't crash
    numeric_cols = [
        'Shipment Volume', 'Tracking %', 'Avg Ping Frequency Mins', 'Milestone Completeness Percent',
        'Origin Arrival Milestones Percent', 'Origin Departure Milestones Percent',
        'Destination Arrival Milestones Percent', 'Destination Departure Milestones Percent',
        'Pickup Arrival Within 30 Min Percent', 'Dropoff Arrival Within 30 Min Percent'
    ]
    for col in numeric_cols:
        df[col] = coerce_numeric(df[col])

    # Only Lane-Carrier rows (case-insensitive)
    df = df[df['Type'].astype(str).str.strip().str.lower() == 'lane-carrier'].copy()

    # Lane label
    df['Lane'] = df['Pickup Location'].astype(str).str.strip() + ' -> ' + df['Dropoff Location'].astype(str).str.strip()

    st.write(f"**Rows (Lane-Carrier only):** {len(df):,} | **Unique Lanes:** {df['Lane'].nunique():,}")

    # Top N lanes by volume
    top_n_lanes = st.number_input("How many top lanes (by volume) to analyze?", min_value=1, value=50, step=1)

    lane_volumes = df.groupby('Lane', as_index=False)['Shipment Volume'].sum()
    top_lanes = lane_volumes.sort_values(by='Shipment Volume', ascending=False).head(top_n_lanes)['Lane']

    suggestions_per_lane = st.number_input("Max number of suggestions per lane", min_value=1, value=3, step=1)
    min_sugg_volume = st.number_input("Minimum shipment volume for a suggested carrier (optional)", min_value=0, value=0, step=10)

    st.markdown("### Select metrics to evaluate and enter thresholds")
    metric_thresholds = {}
    selected_metrics = []

    cols = st.columns(2)
    left_metrics = list(metric_info.items())[:len(metric_info)//2]
    right_metrics = list(metric_info.items())[len(metric_info)//2:]

    with cols[0]:
        for i, (label, info) in enumerate(left_metrics, start=1):
            if st.checkbox(f"{i}. {label} ({'Higher' if info['direction']=='higher' else 'Lower'} is better)", key=f"m_{label}"):
                val = st.number_input(f"Threshold for {label}", key=f"thr_{label}")
                metric_thresholds[label] = val
                selected_metrics.append(label)
    with cols[1]:
        start_idx = len(left_metrics) + 1
        for j, (label, info) in enumerate(right_metrics, start=start_idx):
            if st.checkbox(f"{j}. {label} ({'Higher' if info['direction']=='higher' else 'Lower'} is better)", key=f"m_{label}"):
                val = st.number_input(f"Threshold for {label}", key=f"thr_{label}")
                metric_thresholds[label] = val
                selected_metrics.append(label)

    do_not_replace_raw = st.text_input("Enter carriers you DO NOT want to replace (comma-separated or 'NA')", "NA")
    do_not_replace = parse_do_not_replace(do_not_replace_raw)

    if st.button("Generate Suggestions"):
        with st.spinner("Crunching suggestions..."):
            suggestions = []
            lane_summary_rows = []

            for lane in top_lanes:
                lane_df = df[df['Lane'] == lane]

                good_carriers = []
                bad_carriers = []

                for _, row in lane_df.iterrows():
                    carrier = row['Carrier Name']
                    fail_flag = False

                    for metric in selected_metrics:
                        col = metric_info[metric]['col']
                        threshold = metric_thresholds[metric]
                        direction = metric_info[metric]['direction']

                        val = row[col]
                        # fail if below (for higher) or above (for lower) or NaN
                        if (direction == 'higher' and (pd.isna(val) or val < threshold)) or \
                           (direction == 'lower' and (pd.isna(val) or val > threshold)):
                            fail_flag = True
                            break

                    if fail_flag:
                        if carrier not in do_not_replace:
                            bad_carriers.append(row)
                    else:
                        if (row['Shipment Volume'] or 0) >= min_sugg_volume:
                            good_carriers.append(row)

                total_carriers_lane = len(lane_df)
                lane_summary_rows.append({
                    "Lane": lane,
                    "Good Carriers (meets thresholds)": len(good_carriers),
                    "Bad Carriers (fails thresholds)": len(bad_carriers),
                    "Total Carriers in Lane": total_carriers_lane,
                    "Good Ratio": (len(good_carriers) / total_carriers_lane) if total_carriers_lane else np.nan
                })

                # Build suggestions
                for bad in bad_carriers:
                    replacements = []
                    for good in good_carriers:
                        reasons = []
                        for metric in selected_metrics:
                            col = metric_info[metric]['col']
                            direction = metric_info[metric]['direction']
                            if compare_metrics(good[col], bad[col], direction):
                                # compute delta to show in reason
                                if direction == 'higher':
                                    delta = (good[col] - bad[col]) if (pd.notna(good[col]) and pd.notna(bad[col])) else np.nan
                                else:
                                    # lower is better: improvement = bad - good
                                    delta = (bad[col] - good[col]) if (pd.notna(good[col]) and pd.notna(bad[col])) else np.nan
                                if pd.notna(delta):
                                    reasons.append(f"{metric} (+{round(float(delta), 1)})")
                                else:
                                    reasons.append(metric)

                        if reasons:
                            replacements.append((good, reasons))

                    # sort by current volume carried by suggested carrier in this lane
                    replacements = sorted(
                        replacements,
                        key=lambda x: (x[0].get('Shipment Volume') if pd.notna(x[0].get('Shipment Volume')) else 0),
                        reverse=True
                    )[:suggestions_per_lane]

                    for rep, reasons in replacements:
                        suggestions.append({
                            'Lane': lane,
                            'Suggested Carrier': rep['Carrier Name'],
                            'Replaces': bad['Carrier Name'],
                            'Reason': ", ".join(reasons),
                            **rep.drop(['Type', 'Pickup Location', 'Dropoff Location', 'Carrier Name', 'Lane']).to_dict(),
                            'Replaced Metrics': bad.drop(['Type', 'Pickup Location', 'Dropoff Location', 'Carrier Name', 'Lane']).to_dict(),
                        })

        # Lane Summary (kept)
        lane_summary_df = pd.DataFrame(lane_summary_rows).sort_values(
            by=["Good Carriers (meets thresholds)", "Total Carriers in Lane"],
            ascending=[False, False]
        )
        if not lane_summary_df.empty:
            top_lane_row = lane_summary_df.iloc[0]
            st.info(
                f"🏆 **Highest good-carrier lane (from analyzed lanes):** "
                f"**{top_lane_row['Lane']}** — **{int(top_lane_row['Good Carriers (meets thresholds)'])} good carrier(s)**"
            )
            with st.expander("See Lane Summary (analyzed lanes)"):
                st.dataframe(lane_summary_df, use_container_width=True)

        # Suggestions UI & Export
        if suggestions:
            out_df = pd.DataFrame(suggestions)

            # Expand Replaced Metrics to flat columns
            if 'Replaced Metrics' in out_df.columns:
                replaced_expanded = pd.json_normalize(out_df['Replaced Metrics'])
                replaced_expanded.columns = [f"Replaced | {c}" for c in replaced_expanded.columns]
                out_df = pd.concat([out_df.drop(columns=['Replaced Metrics']), replaced_expanded], axis=1)

            # ---------------------------
            # Dynamic column ordering (YOUR FINAL SPEC)
            # A-D fixed: Lane, Suggested Carrier, Replaces, Reason
            # E fixed: Shipment Volume (suggested)
            # F fixed: Replaced | Shipment Volume
            # Then for each selected metric: suggested col, then Replaced | col
            # Then the rest of the columns in current order
            # ---------------------------
            base_fixed = ['Lane', 'Suggested Carrier', 'Replaces', 'Reason']
            fixed_after = ['Shipment Volume', 'Replaced | Shipment Volume']

            # Build metric pairs in selected order
            metric_pairs = []
            for m in selected_metrics:
                col = metric_info[m]['col']
                sugg_col = col
                rep_col = f"Replaced | {col}"
                if sugg_col in out_df.columns:
                    metric_pairs.append(sugg_col)
                if rep_col in out_df.columns:
                    metric_pairs.append(rep_col)

            already = set(base_fixed + fixed_after + metric_pairs)
            remaining = [c for c in out_df.columns if c not in already]

            ordered_cols = base_fixed + fixed_after + metric_pairs + remaining
            out_df = out_df[ordered_cols]

            st.success(f"✅ Generated {len(out_df):,} suggestions.")
            st.dataframe(out_df, use_container_width=True)

            # CSV download
            csv = out_df.to_csv(index=False)
            st.download_button("Download Suggestions as CSV",
                               data=csv,
                               file_name="carrier_suggestions.csv",
                               mime="text/csv")

            # Excel download (Suggestions + Run_Parameters + Lane_Summary with definitions)
            params_rows = [
                {
                    'Parameter': 'Top N Lanes',
                    'Value': int(top_n_lanes),
                    'Definition': 'Only these highest-volume lanes are analyzed for suggestions.'
                },
                {
                    'Parameter': 'Suggestions per Lane',
                    'Value': int(suggestions_per_lane),
                    'Definition': 'Maximum number of suggested carriers to output per lane (ranked by suggested carrier volume).'
                },
                {
                    'Parameter': 'Min Suggested Volume',
                    'Value': int(min_sugg_volume),
                    'Definition': 'Suggested carriers must have at least this many shipments in the lane to be considered.'
                },
                {
                    'Parameter': 'Do Not Replace',
                    'Value': ", ".join(do_not_replace) if do_not_replace else "NA",
                    'Definition': 'Carriers listed here will not be considered for replacement even if they fail thresholds.'
                },
                {
                    'Parameter': 'Selected Metrics',
                    'Value': ", ".join(selected_metrics),
                    'Definition': 'Metrics chosen to judge good/bad carriers and to compare suggested vs. replaced.'
                },
                {
                    'Parameter': 'Thresholds',
                    'Value': ", ".join(f"{m}:{metric_thresholds[m]}" for m in selected_metrics),
                    'Definition': 'For each selected metric: Higher is better except Avg Ping Frequency Mins (lower is better).'
                }
            ]
            params_df = pd.DataFrame(params_rows)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                out_df.to_excel(writer, sheet_name="Suggestions", index=False)
                params_df.to_excel(writer, sheet_name="Run_Parameters", index=False)
                lane_summary_df.to_excel(writer, sheet_name="Lane_Summary", index=False)
            st.download_button(
                "Download Suggestions as Excel",
                data=buffer.getvalue(),
                file_name="carrier_suggestions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No suggestions found based on the current thresholds and selections.")
