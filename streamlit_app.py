import streamlit as st
import pandas as pd
import io

# ------------------------
# Utility Functions
# ------------------------
def parse_do_not_replace(value):
    if value.strip().lower() == 'na':
        return []
    return [x.strip() for x in value.split(',') if x.strip()]

def compare_metrics(sugg_val, replace_val, direction):
    try:
        if pd.isnull(sugg_val) or pd.isnull(replace_val):
            return False
        if direction == 'higher':
            return sugg_val > replace_val
        else:
            return sugg_val < replace_val
    except Exception as e:
        return False

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
st.title("🚚 Carrier Suggestion Tool for Truckload Lanes")

uploaded_file = st.file_uploader("Upload the CSV file", type=["csv"])

if uploaded_file:
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

    df = df[df['Type'] == 'Lane-Carrier']  # Only Lane-Carrier rows
    df['Lane'] = df['Pickup Location'] + ' -> ' + df['Dropoff Location']

    # Top N lanes by volume
    top_n_lanes = st.number_input("How many top lanes (by volume) to analyze?", min_value=1, value=50)

    lane_volumes = df.groupby('Lane')['Shipment Volume'].sum().reset_index()
    top_lanes = lane_volumes.sort_values(by='Shipment Volume', ascending=False).head(top_n_lanes)['Lane']

    suggestions_per_lane = st.number_input("Max number of suggestions per lane", min_value=1, value=3)

    st.markdown("### Select metrics to evaluate and enter thresholds")
    metric_thresholds = {}
    selected_metrics = []

    for i, (label, info) in enumerate(metric_info.items(), start=1):
        if st.checkbox(f"{i}. {label} ({'Higher' if info['direction']=='higher' else 'Lower'} is better)"):
            val = st.number_input(f"Threshold for {label}", key=label)
            metric_thresholds[label] = val
            selected_metrics.append(label)

    do_not_replace_raw = st.text_input("Enter carriers you DO NOT want to replace (comma-separated or 'NA')", "NA")
    do_not_replace = parse_do_not_replace(do_not_replace_raw)

    if st.button("Generate Suggestions"):
        suggestions = []

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
                    if (direction == 'higher' and val < threshold) or (direction == 'lower' and val > threshold):
                        fail_flag = True
                        break

                if fail_flag:
                    if carrier not in do_not_replace:
                        bad_carriers.append(row)
                else:
                    good_carriers.append(row)

            for bad in bad_carriers:
                replacements = []
                for good in good_carriers:
                    reasons = []
                    for metric in selected_metrics:
                        col = metric_info[metric]['col']
                        direction = metric_info[metric]['direction']
                        if compare_metrics(good[col], bad[col], direction):
                            reasons.append(metric)

                    if reasons:
                        replacements.append((good, reasons))

                replacements = sorted(replacements, key=lambda x: x[0]['Shipment Volume'], reverse=True)[:suggestions_per_lane]

                for rep, reasons in replacements:
                    suggestions.append({
                        'Lane': lane,
                        'Suggested Carrier': rep['Carrier Name'],
                        'Replaces': bad['Carrier Name'],
                        'Reason': ", ".join(reasons),
                        **rep.drop(['Type', 'Pickup Location', 'Dropoff Location', 'Carrier Name', 'Lane']).to_dict(),
                        'Replaced Metrics': bad.drop(['Type', 'Pickup Location', 'Dropoff Location', 'Carrier Name', 'Lane']).to_dict(),
                    })

        if suggestions:
            out_df = pd.DataFrame(suggestions)
            st.success(f"Generated {len(out_df)} suggestions.")
            st.dataframe(out_df)

            # Download link
            csv = out_df.to_csv(index=False)
            st.download_button("Download Suggestions as CSV", data=csv, file_name="carrier_suggestions.csv", mime="text/csv")
        else:
            st.warning("No suggestions found based on the current thresholds and selections.")
